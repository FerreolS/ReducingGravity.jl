


function gravi_compute_wavelength_bounds(spectra::Dict{String, ConcreteWeightedData{T,N}},
	profiles::Dict{String,SpectrumModel{A,B,C,D,E}},
	thrs=0.01,
	kwds...) where {T,N,A,B,C,D,E} 
	
	if all(isfinite,[p.λbnd[2] for (_,p) ∈ profiles])
		return profiles,spectra
	end
	pr_array = Vector{Pair{String,SpectrumModel{A,B,C,D,E}}}(undef,length(profiles))
	spectra_array = Vector{Pair{String,ConcreteWeightedData{T,N}}}(undef,length(spectra))
	Threads.@threads for (i,(key,profile)) ∈ collect(enumerate(profiles) )
		tel1 = key[1] 
		tel2 = key[2]
		key1 = "$tel1-$key" 
		key2 = "$tel2-$key" 
		
		wvlngth = get_wavelength(profile)
		
		thrs1 = median(spectra[key1].val) .* thrs
		thrs2 = median(spectra[key2].val) .* thrs
		
		pixmin = min(findfirst(spectra[key1].val .>= thrs1),findfirst(spectra[key2].val .>= thrs2))
		pixmax = max(findlast(spectra[key1].val .>= thrs1),findlast(spectra[key2].val .>= thrs2))
		
		λmin = wvlngth[pixmin]
		λmax = wvlngth[pixmax]
		@reset profile.λbnd = [λmin, λmax]
		pr_array[i] = key=>profile
		spectra_array[2*i-1] = key1 => WeightedData(spectra[key1],pixmin:pixmax)
		spectra_array[2*i] 	 = key2 => WeightedData(spectra[key2],pixmin:pixmax)
	end
	return Dict(pr_array),Dict(spectra_array)
	
end


function gravi_compute_transmissions(  spectra::Dict{String, ConcreteWeightedData{T,N}},
										profiles::Dict{String,SpectrumModel{A,B,C,D,T}},
										lamp;
										thrs=0.01,
										kwds...) where {T,N,A,B,C<:Interpolator,D} 
	if (D <: Number) 
		pr_array = Vector{Pair{String,SpectrumModel{A,B,C,Vector{T},T}}}(undef,length(profiles))
	else
		pr_array = Vector{Pair{String,SpectrumModel{A,B,C,D,T}}}(undef,length(profiles))
	end		
	Threads.@threads for (i,(key,profile)) ∈ collect(enumerate(profiles) )
		tel1 = key[1] 
		tel2 = key[2]
		key1 = "$tel1-$key" 
		key2 = "$tel2-$key" 
		BSp1 = profile.transmissions[1].basis
		BSp2 = profile.transmissions[2].basis


		wvlngth =  T.(get_wavelength(profile; bnd=false))
		lmp =lamp.(wvlngth)
		good = isfinite.(lmp) .&& (lmp .!= 0)
		wvgood = wvlngth[good]
		lmp = view(lmp,good)
		transmissions = [gravi_fit_transmission( view(spectra[key1],good),lmp,BSp1,wvgood; kwds...)
						gravi_fit_transmission( view(spectra[key2], good),lmp,BSp2, wvgood; kwds...)]
		@reset profile.transmissions = transmissions

		#flat = ones(T,length(spectra[key1]))
		flat  = ((view(spectra[key1],good) / (profile.transmissions[1].(wvgood).* lmp) + view(spectra[key2],good) / (profile.transmissions[2].(wvgood).* lmp))/2).val
		flat[flat.==0] .= 1
		@reset profile.flat = flat
		pr_array[i] = key=>profile
	end
	profiles = Dict(pr_array)

	return profiles

end

function gravi_init_transmissions(profiles::Dict{String,SpectrumModel{A,B,C,D,E}};
									T=Float64,
									nb_transmission_knts=20,
									kernel = CatmullRomSpline{T}(),
									kwds...
									) where {A,B,C,D,E} 
									
	λmin = minimum([max(p.λbnd[1],	minimum(filter!(!isnan,get_wavelength(p)))) for p ∈ values(profiles)])
	λmax = maximum([min(p.λbnd[2],	maximum(filter!(!isnan,get_wavelength(p)))) for p ∈ values(profiles)])
	knt =  range(T(λmin),T(λmax),nb_transmission_knts)



	pr_array = Vector{Pair{String,SpectrumModel{A,B,Interpolator{typeof(knt),typeof(kernel)},D,T}}}(undef,length(profiles))
	for (i,(key,profile)) ∈ collect(enumerate(profiles) )
		λ = get_wavelength(profile)
		λ = λ[isfinite.(λ)]
		S = Interpolator(knt,kernel)
		initcoefs = compute_coefs(S,λ, ones(T,length(λ)))
		@reset profile.transmissions = [InterpolatedSpectrum(copy(initcoefs),S)
										InterpolatedSpectrum(copy(initcoefs),S)]
		pr_array[i] = key=>profile
	end
	profiles = Dict(pr_array)
	return profiles
end

function gravi_fit_transmission(	spectrum::A,
									lampspectrum,
									B::Interpolator,
									wavelength;
									Chi2=  1,
									kwd...) where {T, A<:AbstractWeightedData{T, 1}} 
	coefs = compute_coefs(B,wavelength,spectrum/lampspectrum; Chi2=Chi2)

	return InterpolatedSpectrum(coefs,B)

end




function gravi_compute_lamp(spectra::Dict{String, S}, 
							profiles::Dict{String,SpectrumModel{A,B,C,D,E}};
							nb_lamp_knts=360,
							init_lamp = nothing,
							kernel = CatmullRomSpline{T}(),
							kwds...) where {T,A,B,C<:Interpolator,D,S<:AbstractWeightedData{T, 1},E} 
	
	data_trans = Vector{@NamedTuple{spectrum::WeightedData{T,1,SubArray{T, 1, Vector{T}, Tuple{Vector{Int64}}, false},SubArray{T, 1, Vector{T}, Tuple{Vector{Int64}}, false}},
									transmission::Vector{T},wavelength::B}}(undef,length(spectra))	
	for (i,(key,spectrum)) ∈ enumerate(spectra)		
		profile = profiles[ key[3:end]]
		transmission = key[1] == key[3] ? profile.transmissions[1] : profile.transmissions[2]
		wvlngth = get_wavelength(profile; bnd=false)
		good = .!isnan.(wvlngth)
		data_trans[i] = (;spectrum = view(spectrum,good), transmission = transmission(wvlngth[good]), wavelength = wvlngth[good])
	end
	local Bs

	if isnothing(init_lamp)
		λmin = minimum([max(p.λbnd[1],	minimum(filter!(!isnan,get_wavelength(p)))) for p ∈ values(profiles)])
		λmax = maximum([min(p.λbnd[2],	maximum(filter!(!isnan,get_wavelength(p)))) for p ∈ values(profiles)])
		knt = range(T.(λmin),T.(λmax),nb_lamp_knts)
		Bs =  Interpolator(knt,kernel)
	else
		Bs = init_lamp.basis
	end
	coefs = gravi_fit_lamp(  data_trans,Bs; kwds...)
	return InterpolatedSpectrum(coefs, Bs)
end

function gravi_fit_lamp(data_trans,(;kernel,knots)::Interpolator)
	#solve A x = b
	# b = sum_i ( H_i' * (w_i .* d_i))
	T = eltype(kernel)
	b = zeros(T,length(knots))
	A = zeros(T,length(knots),length(knots))
	Threads.@threads for (;spectrum,transmission, wavelength) ∈ data_trans
		K = build_interpolation_matrix(kernel,knots,wavelength) .* transmission
		b .+= K' * (spectrum.precision .* spectrum.val )
		A .+= K'* (spectrum.precision .* K)
	end
	F = cholesky(A; check=false)
    if issuccess(F)
        return   F \ b
    else
        return Symmetric(pinv(A)) * b
    end
end



function gravi_extract_profile_flats_from_p2vm(	P2VM::Dict{String, ConcreteWeightedData{T, 2}}, 
												dark::AbstractWeightedData{T,N},
												profiles::AbstractDict; 
												kwds...
											) where {T,N}

	spctr = Dict{String,ConcreteWeightedData{T,1}}()
	
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

function gravi_extract_profile_flats_from_p2vm(	flats::Vector{W}, 
												chnames::Matrix{String} ,
												profiles::AbstractDict; 
												kwds...
											) where {T,W<:AbstractWeightedData{T, 2}}


	uniqname = unique(chnames)
	spctr = Vector{Pair{String,ConcreteWeightedData{T,1}}}(undef,length(uniqname))
	
	for (i , chnl) ∈ enumerate(uniqname)
		idx = [idxt[1] for idxt ∈ findall( x -> x == chnl,  chnames)]
		profile = profiles[chnl[3:end]]
		ch1 = gravi_extract_profile(flats[idx[1]] ,profile;kwds...)
		ch2 = gravi_extract_profile(flats[idx[2]] ,profile;kwds...)
		spctr[i] = chnl => combine(ch1,ch2)
	end
	return Dict(spctr)

end


function gravi_compute_gain_from_p2vm(	P2VM::Dict{String, ConcreteWeightedData{T, 2}}, 
										profiles::AbstractDict,
										goodpix::BitMatrix; 
										restrict=0.0, 
										substract_dark=true,
										fix_gain=true,
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
	#return avg,prec
	usable = ill .& goodpix  .& reduce(.&,prec .!=0, dims=3,init=true)[:,:,1]
	gain, rov = build_ron_and_gain(usable,avg,prec; substract_dark=substract_dark,fix_gain=fix_gain)
	darkp2vm = WeightedData(avg[:,:,5],prec[:,:,5])
	#darkp2vm = WeightedData(avg,prec)

	return darkp2vm,gain, rov

end


function build_ron_and_gain(usable::BitMatrix,
							avg::Array{T,3},
							prec::Array{T,3}; 
							substract_dark=true,
							fix_gain=true) where T
	sz = size(usable)
	gain = zeros(T,sz[1])
	rov = zeros(T,sz[1])
	for i ∈ axes(usable,1)
		u = findall(usable[i,:] )
		if substract_dark
			a = (view(avg[i,:,:],u,:) .- view(avg[i,:,:],u,5))[:]
			v = (1 ./ view(prec[i,:,:],u,:) .+ 1 ./ view(prec[i,:,5],u))[:]
		else 
			a = (view(avg[i,:,:],u,:))[:]
			v = (1 ./ view(prec[i,:,:],u,:))[:]
		end
		P = hcat(ones(length(a)), a)	
		rov[i], gain[i] = inv(P'*P) * P' * v
		#rov[i], gain[i] = ldiv!(cholesky!(Symmetric(P'*P)),P'*v )
	end

	mgain = median(gain)
	sgain = mad(gain)
	if fix_gain
		gain[.!((-3*sgain) .< (gain .- mgain) .< (3*sgain))] .= mgain
	end
	return 1 ./ gain, rov
end

function gravi_compute_gain_from_p2vm(	flats::Vector{C}, 
										dark::C,
										profiles::AbstractDict,
										goodpix::BitMatrix; 
										restrict=0.0, 
										fix_gain=true,
										substract_dark = false,
										kwds...
										) where {T,C<:ConcreteWeightedData{T, 2}}
											
	sz = size(dark)
	avg = Array{T,3}(undef,sz...,length(flats)+1)
	prec = Array{T,3}(undef,sz...,length(flats)+1)
	
	for (i,flat) ∈ enumerate(flats)
		avg[:,:,i] = flat.val
		prec[:,:,i] = flat.precision
	end
	
	avg[:,:,end] = dark.val
	prec[:,:,end] = dark.precision


	ill = falses(sz)
	for profile ∈ values(profiles )

		bbox = profile.bbox

		if restrict>0
			model =  get_profile(profile, bbox)
			bbox = bbox[model .> restrict]
		else
			bbox = bbox[:]
		end
		
		view(ill,bbox) .= true
				
	end
	usable = ill .& goodpix  .& reduce(.&,prec .!=0, dims=3,init=true)[:,:,1]
	gain, rov = build_ron_and_gain(usable,avg,prec; substract_dark=substract_dark,fix_gain=fix_gain)

	return gain, rov

end

 

function gravi_compute_flat_and_dark_from_p2vm(	P2VM::Dict{String, Array{T, 3}}, 
										bboxes::Dict{String,C},
										illuminated::BitMatrix,
										goodpix::BitMatrix; 
										filterblink=true,
										unbiased=true, 
										kwds...
											) where {T,C}
											
	sz = size(first(values(P2VM)))
	sorted = Array{T,4}(undef,sz...,6)
	ind = ones(Int,length(bboxes))

	for (i,(key,(_,bbox))) ∈ enumerate(bboxes )


		
		
		for (baseline,data) ∈ P2VM
			tel1,tel2 = baseline[5] , baseline[6]
			#tel1,tel2 = baseline[5] < baseline[6] ? (baseline[5] , baseline[6]) : (baseline[6] , baseline[5])

		
			t1,t2 = key[1] , key[2] 
			#t1,t2 = key[1] < key[2] ? (key[1],key[2]) : (key[2],key[1] )
			
			ill1 = (t1 == tel1) || (t1 == tel2) 
			ill2 = (t2 == tel1) || (t2 == tel2) 
			if (ill1 && ill2 ) # interferometric channel
				idx = 6
			elseif (!ill1 && !ill2)  # non illuminated channel
				idx =5
			else 
				idx =  ind[i]
				ind[i] += 1
			end 
			#avgbias, data = gravi_data_detector_cleanup(data,illuminated)
			view(sorted,bbox,:,idx)  .= view(data,bbox,:)
			
		end
		
	end
	#return sorted
	wd = Vector{ConcreteWeightedData{T,2}}(undef,5)
	gp = Vector{BitMatrix}(undef,5)
	Threads.@threads for i∈1:5
		wd[i], gp[i] = gravi_create_weighteddata(sorted[:,:,:,i],illuminated,goodpix,filterblink=filterblink,unbiased=unbiased,keepbias=true,cleanup=false)
	end
	#goodpix .&= reduce(.&,gp)
	goodpix .&= gp[5]
	Threads.@threads for i∈1:5
		flagbadpix!(wd[i],.!goodpix)
	end
	
	return wd[1:4], wd[5], sorted[:,:,:,6],goodpix

end


function gravi_reorder_p2vm(	P2VM::Dict{String, Array{T, 3}}, 
								bboxes::Dict{String,C},
								illuminated::BitMatrix,
								goodpix::BitMatrix; 
								keepbias=true,
								filterblink=true,
								blinkkernel=5,
								unbiased=true, 
								kwds...
											) where {T,C}
											
	sz = size(first(values(P2VM)))
	sorted = Array{T,4}(undef,sz...,6)
	ind = ones(Int,length(bboxes))
	chnames = Matrix{String}(undef,4,length(bboxes))


	Abias = Array{Matrix{T}}(undef,6)
	for (i, (key,data)) ∈ enumerate(P2VM)
		Abias[i] = gravi_data_detector_cleanup!(data,illuminated;keepbias=keepbias)
		
	end
	bias = mean(Abias)

	for (baseline,data) ∈ P2VM
		tel1,tel2 = baseline[5] , baseline[6]
		for (i,(key,(_,bbox))) ∈ enumerate(bboxes )
			t1,t2 = key[1] , key[2] 
			#tel1,tel2 = baseline[5] < baseline[6] ? (baseline[5] , baseline[6]) : (baseline[6] , baseline[5])

		
			#t1,t2 = key[1] < key[2] ? (key[1],key[2]) : (key[2],key[1] )
			
			ill1 = (t1 == tel1) || (t1 == tel2) 
			ill2 = (t2 == tel1) || (t2 == tel2) 
			if (ill1 && ill2 ) # interferometric channel
				idx = 6
			elseif (!ill1 && !ill2)  # non illuminated channel
				idx =5
			else 
				idx =  ind[i]
				chnames[idx,i] = (ill1 ? "$t1-$key" : "$t2-$key")

				ind[i] += 1
			end 
			#avgbias, data = gravi_data_detector_cleanup(data,illuminated)
			view(sorted,bbox,:,idx)  .= view(data,bbox,:)
			
		end
		
	end
	#return sorted
	wd = Vector{ConcreteWeightedData{T,2}}(undef,5)
	gp = Vector{BitMatrix}(undef,5)
	Threads.@threads for i∈1:5
		wd[i], gp[i] = gravi_create_weighteddata(sorted[:,:,:,i],illuminated,goodpix;cleanup = false,filterblink=filterblink,unbiased=unbiased,blinkkernel=blinkkernel,bias=bias)
	end
	#goodpix .&= reduce(.&,gp)
	goodpix .&= gp[5]
	Threads.@threads for i∈1:5
		flagbadpix!(wd[i],.!goodpix)
	end
	
	return wd[1:4], wd[5], sorted[:,:,:,6],goodpix,chnames

end
