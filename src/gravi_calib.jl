
function fitprofile(data::AbstractWeightedData{T,2},bndbx::C; center_degree=4, σ_degree=4, thrsld=0.1) where{T,C<:CartesianIndices}

	fulldata = view(data,bndbx)
	spectra = (sum(fulldata.val .* fulldata.precision,dims=2)./ sum(fulldata.precision,dims=2))[:]
	firstidx = findfirst(x -> x>mean(spectra)*thrsld,spectra)
	lastidx = findlast(x -> x>mean(spectra)*thrsld,spectra)


	data = view(data,bndbx[firstidx:lastidx,:])
	
	specmodel = ProfileModel(bndbx[firstidx:lastidx,:];maxdeg = max(center_degree,σ_degree), precond=true)

	#shp = (sum(data .* wght,dims=1)./ sum(wght,dims=1))[:]


	center = zeros(center_degree+1)
	σ = zeros(σ_degree+1)
	center[1] = mean(bndbx.indices[2])
	
	σ[1] = 0.5 #std((shp .* ay) ./ sum(shp))
	θ = (;center=center, σ = σ)
	params, unflatten = destructure(θ)
	f(params) = scaledlikelihood(data,specmodel(;unflatten(params)...))

	res = optimize(f, params, NelderMead(),Optim.Options(iterations=10000))
	xopt = Optim.minimizer(res)
	θopt= unflatten(xopt)
	(;center,σ) = θopt
	center .*=  specmodel.preconditionner[1:σ_degree+1]
	σ .*=  specmodel.preconditionner[1:σ_degree+1]
	θopt = (;center=center,σ=σ)
	return θopt
end


function gravi_extract_profile(	data::AbstractWeightedData{T,N},
								profile::SpectrumModel; 
								restrict=0.01, 
								nonnegative=false, 
								robust=false) where {T,N}
	bbox = profile.bbox
	(;val, precision) = view(data,bbox)

	model =  get_profile(profile)
	if restrict>0
		model .*= (model .> restrict)
	end

	αprecision =sum(  model.^2 .* precision ,dims=2)[:]
	α = sum(model .* precision .* val,dims=2)[:] ./ αprecision
	nanpix = .! isnan.(α)
	if nonnegative
		positive = nanpix .& (α .>= T(0))
	else
		positive = nanpix
	end
	wd = WeightedData(positive .* α, positive .* αprecision)

	if robust # Talwar hard descender
		res = sqrt.(precision) .* (wd.val  .* model .- val) 
		
		good = (T(-2.795) .< res .<  T(2.795))
		αprecision =sum( good .* model.^2 .* precision ,dims=2)[:]
		α = sum(good .* model .* precision .* val,dims=2)[:] ./ αprecision
		
		nanpix = .! isnan.(α)
		if nonnegative
			positive = nanpix .& (α .>= T(0))
		else
			positive = nanpix
		end
		wd = WeightedData(positive .* α, positive .* αprecision)
	end
	return wd
end

function gravi_extract_profile(	data::AbstractWeightedData{T,N},	
								profile::Dict{String,<:SpectrumModel}; 
								kwds...) where {T,N}
	profiles = Dict{String,AbstractWeightedData{Float64,1}}()
	for (key,val) ∈ profile
		push!(profiles,key=>gravi_extract_profile(data ,val; kwds...))
	end
	return profiles
end

function gravi_extract_profile_flats(	flats::Vector{<:AbstractWeightedData{T,N}},
										profile::Dict{String,<:SpectrumModel}; 
										kwds...) where {T,N}
	spctr = Dict{String,AbstractWeightedData{Float64,1}}()
	Threads.@threads for tel1 ∈ 1:4
		for tel2 ∈ 1:4
			tel1==tel2 && continue
			for chnl ∈ ["A","B","C","D"]
				haskey(profile,"$tel1$tel2-$chnl-C") || continue
				prfl =profile["$tel1$tel2-$chnl-C"] 
				push!(spctr,"$tel1-$tel1$tel2-$chnl-C"=>gravi_extract_profile(flats[tel1],prfl; kwds...))
				push!(spctr,"$tel2-$tel1$tel2-$chnl-C"=>gravi_extract_profile(flats[tel2],prfl; kwds...))
			end
		end
	end
	return spctr
end

function gravi_extract_profile(	data::AbstractArray{T,N},
								precision::Union{BitMatrix,AbstractArray{T,N}},
								profile::Dict{String,<:SpectrumModel}; 
								kwds...) where {T,N}
	profiles = Dict{String,AbstractWeightedData{T,1}}()
	for (key,val) ∈ profile
		push!(profiles,key=>gravi_extract_profile(data,precision ,val; kwds...))
	end
	return profiles
end

function gravi_extract_profile(	data::AbstractArray{T,N},
								precision::Union{BitMatrix,AbstractArray{T,N}},
								profile::SpectrumModel; 
								restrict=0.01, 
								nonnegative=false, 
								robust=false) where {T,N}
	bbox = profile.bbox
	if N==2
		val= view(data,bbox)
		prec= view(precision,bbox)
	else
		val= view(data,bbox,:)
		prec= view(precision,bbox,:)
	end
	model =  get_profile(profile)
	if restrict>0
		model .*= (model .> restrict)
	end

	αprecision =sum(  model.^2 .* prec ,dims=2)[:]
	α = sum(model .* prec .* val,dims=2)[:] ./ αprecision
	nanpix = .! isnan.(α)
	if nonnegative
		positive = nanpix .& (α .>= T(0))
	else
		positive = nanpix
	end
	return	positive .* α

end




function gravi_compute_gain(	flats::Vector{<:AbstractArray{T,3}},
								illuminated::BitMatrix,
								goodpix::BitMatrix,
								profiles::Dict{String,<:SpectrumModel}; 
								restrict=0.01, 
								thrsld=0.1,
								nonnegative=false,  
								filterblink=true,
								unbiased=true) where {T}
	
	
	S2 = Vector{T}()
	#VarS2 = Vector{T}()
	Avg = Vector{T}()
	goodpix = copy(goodpix)
	
	 for tel1 ∈ 1:4
		rawdata = flats[tel1]
		bias, flatdata = gravi_data_detector_cleanup(rawdata,illuminated)
		
		spectra = sum(flatdata.*goodpix, dims=(2,3))[:] 
		firstidx = findfirst(x -> x>mean(spectra)*thrsld,spectra)
		lastidx = findlast(x -> x>mean(spectra)*thrsld,spectra)
		
		
		if filterblink
			blink = gravi_compute_blink(flatdata,bias=bias)
			goodpix .&= (sum(blink,dims=3) .> max(3,0.75 * size(blink,3)))
		else
			blink = trues(size(flatdata))
		end
		for tel2 ∈ 1:4
			tel1==tel2 && continue
			for chnl ∈ ["A","B","C","D"]
				haskey(profiles,"$tel1$tel2-$chnl-C") || continue
				profile =profiles["$tel1$tel2-$chnl-C"] 
				bbox = profile.bbox[firstidx:lastidx,:]
				data = view(flatdata,bbox,:)
				gpblink = view(goodpix,bbox).*view(blink,bbox,:)
				
				Nobs = sum(gpblink, dims=3)[:,:,1]
				avg =  (sum(data.*gpblink, dims=3)[:,:,1] ./ Nobs)

				
				if restrict>0
					model =  get_profile(profile, bbox)
					restricted = (model .> restrict) .&& (Nobs .>3) 
				else
					restricted =(Nobs .>3)
				end
				Nobs =Nobs[restricted]

				push!(Avg, avg[restricted]...)
				push!(S2 , (sum(gpblink.*(data .- avg).^2,dims=3)[:,:,1][restricted] ./ (Nobs .- 1))...)
				
				#μ4 =  sum(gpblink .*(data .- avg).^4, dims=3)[:,:,1][restricted] ./ Nobs
				#μ2 =  sum(gpblink .*(data .- avg).^2, dims=3)[:,:,1][restricted] ./ Nobs
				#push!(VarS2 , (μ4 ./ Nobs .- (Nobs .-3) ./ (Nobs .* (Nobs .-1)) .* μ2.^2)...)
			end
		end
	end
	P = hcat(ones(size(Avg)), Avg)
	wRG = inv(P'*( P ))
	ron, gain = wRG * P' * (S2 )
	return sqrt(ron),1/gain
end

function gravi_fit_transmission(	spectrum::AbstractWeightedData{T, 1},
									lampspectrum::Vector{Float64},
									initcoefs::Vector{Float64},
									B;
									verb=0,
									maxeval=50, 
									kwd...) where T 
	rng= 1:length(lampspectrum)
	function loss(rng,spectrum,B,lampspectrum,coefs) 
		S = Spline(B,coefs)
		return likelihood(spectrum,map(S,rng) .* lampspectrum)
	end
	f(x) = loss(rng,spectrum,B,lampspectrum,x)
	coefs = vmlmb(f,  initcoefs ;autodiff=true, verb=verb,maxeval=maxeval,kwd...)
	
	return Transmission(coefs,B)

end


function gravi_compute_transmission_and_lamp(	spectra::Dict{String, AbstractWeightedData{T, 1}};  
										verb=10,maxeval=50, kwd...) where T 
	spectraArray =values(spectra)
	nspectra = length(spectra)
	meanspectrum = sum(spectraArray) / nspectra
	rng= 1:length(meanspectrum)
	knt = SVector{18,Float32}(1.0, 24.0, 35.0, 41.0, 46.0, 58.0, 69.0, 91.0, 114.0, 125.0, 136.0, 159.0, 181.0, 226.0, 271.0, 294.0, 316.0, 360.0)
	#sp4 = Spline1D(1:360, meanspectrum.val; w=meanspectrum.precision, k=3, bc="zero",s=0.01)
	B = BSplineBasis(BSplineOrder(3), knt)
	ncoefs = length(B)
	coefs = [ [zeros(Float64,3)...,ones(Float64,ncoefs-6)...,zeros(Float64,3)...] for i ∈ 1:nspectra]
	#coefs = [ones(T,ncoefs) for i ∈ 1:nspectra]
	lamp = meanspectrum.val

	
	x = (;coefs=coefs)
	x0, restructure = destructure(x)
	function loss(rng,spectraArray,B;coefs::Vector{<:Vector{T1}}= coefs, lamp::Vector{T2}=lamp) where{T1,T2}
		S = Spline.(B,coefs)
		return mapreduce((x,y)->likelihood(y,map(x,rng) .* lamp)::promote_type(T1,T2),+,S,spectraArray)
	end
	f(x) = loss(rng,spectraArray,B;restructure(x)...)
	xopt = vmlmb(f,  x0 ;autodiff=true, verb=verb,maxeval=maxeval,kwd...)
	(;coefs) = restructure(xopt)

	transmissions = Dict(val=>Transmission(coefs[i],B) for (i,val) ∈ enumerate(keys(spectra)))
	return (transmissions, lamp)

end

function gravi_extract_profile_flats_from_p2vm(	P2VM::Dict{String, WeightedData{T, 2,A,B}}, 
												dark::AbstractWeightedData{T,N},
												profiles::Dict{String,<:SpectrumModel}; 
												kwds...
											) where {T,N,A,B}

	spctr = Dict{String,ConcreteWeightedData{Float64,1}}()
	
	for (baseline,data) ∈ P2VM
		tel1 = baseline[5]
		tel2 = baseline[6]
	
		for (key,profile) ∈ profiles 
			t1 = key[1] 
			t2 = key[2]
			
			ill1 = (t1 == tel1) || (t1 == tel2) 
			ill2 = (t2 == tel1) || (t2 == tel2) 
			(ill1 && ill2 ) && continue # interferometric channel
			(!ill1 && !ill2) && continue # non illuminated channel
			name = (ill1 ? "$t1-$key" : "$t2-$key") 

			pr = gravi_extract_profile(data - dark,profile;kwds...)
			if haskey(spctr,name)
				pr = (pr +  spctr[name])/2
			end 
			#sptr_array[i] = name=>pr
			push!(spctr,name=>pr)
		end
		
	end
	#spctr = Dict(sptr_array)
	return spctr
	

end


function gravi_compute_gain_from_p2vm(	P2VM::Dict{String, ConcreteWeightedData{T, 2}}, 
										profiles::Dict{String,<:SpectrumModel},
										goodpix::BitMatrix; 
										restrict=0.0, 
										kwds...
											) where {T}
											
	sz = size(first(values(P2VM)))
	avg = Array{T,3}(undef,sz...,5)
	prec = Array{T,3}(undef,sz...,5)
	ill = falses(sz)
	ind = ones(Int,length(profiles))

	for (i,(key,profile)) ∈ enumerate(profiles )

		bbox = profile.bbox

		if restrict>0
			model =  get_profile(profile, bbox)
			bbox = bbox[model .> restrict]
		else
			bbox = bbox[:]
		end
		
		for (baseline,data) ∈ P2VM
			tel1,tel2 = baseline[5] , baseline[6]
			#tel1,tel2 = baseline[5] < baseline[6] ? (baseline[5] , baseline[6]) : (baseline[6] , baseline[5])

		
			t1,t2 = key[1] , key[2] 
			#t1,t2 = key[1] < key[2] ? (key[1],key[2]) : (key[2],key[1] )
			
			ill1 = (t1 == tel1) || (t1 == tel2) 
			ill2 = (t2 == tel1) || (t2 == tel2) 
			(ill1 && ill2 ) && continue # interferometric channel
			if (!ill1 && !ill2)  # non illuminated channel
				idx =5
			else 
				idx =  ind[i]
				ind[i] += 1
			end 
			
			d = view(data,bbox)
			view(avg,bbox,idx)  .= d.val[:]
			view(prec,bbox,idx) .= d.precision[:]
			view(ill,bbox) .= true
		end
		
	end
	usable = ill .& goodpix  .& reduce(.&,prec .!=0, dims=3,init=true)[:,:,1]
	gain, rov = build_ron_and_gain(usable,avg,prec)
	darkp2vm = WeightedData(avg[:,:,5],prec[:,:,5])
	return darkp2vm,gain, rov

end


function build_ron_and_gain(usable::BitMatrix,avg::Array{T,3},prec::Array{T,3}) where T
	sz = size(usable)
	gain = zeros(T,sz[1])
	rov = zeros(T,sz[1])
	for i ∈ axes(usable,1)
		u = findall(usable[i,:] )
		a = (view(avg[i,:,:],u,:) .- view(avg[i,:,:],u,5))[:]
		v = (1 ./ view(prec[i,:,:],u,:) .+ 1 ./ view(prec[i,:,5],u))[:]
		P = hcat(ones(length(a)), a)	
		rov[i], gain[i] = inv(P'*P) * P' * v
	end
	return 1 ./ gain, rov
end


#= 
function test(sm::SpectrumModel{T};center=[0.0],fwhm=[1.0],amplitude=[1.0]) where {T}
	out=Matrix{T}(undef,length(sm.ax), length(sm.ay))
	for (iy,y) ∈ enumerate(sm.ay)
		c,f,amp=  zeros(T,3)
		
		for (center_index,center_coefs) ∈ enumerate(center)
			c += T(sm.preconditionner[center_index]*center_coefs .* y.^ (center_index-1))
		end
		for (fwhm_index,fwhm_coefs) ∈ enumerate(fwhm)
			f += T(sm.preconditionner[fwhm_index]*fwhm_coefs .* y.^ (fwhm_index-1))
		end
		for (amp_index,amp_coefs) ∈ enumerate(amplitude)
			amp += T(sm.preconditionner[amp_index]*amp_coefs .* y.^(amp_index-1))
		end
		halfprecision = inv((T(2/ (2 * sqrt(2 * log(2.))))*f)^2)

		for (ix,x) ∈ enumerate(sm.ax)
			out[ix,iy] = amp * exp(-(x - c)^2 * halfprecision)
		end
	end
	return out
end =#
 