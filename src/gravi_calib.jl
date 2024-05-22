


function gravi_compute_wavelength_bounds(spectra::Dict{String, ConcreteWeightedData{T,N}},
										profiles::Dict{String,SpectrumModel{A,B,C}},
										thrs=0.01,
										kwds...) where {T,N,A,B,C} 
	pr_array = Vector{Pair{String,SpectrumModel{A,B,C}}}(undef,length(profiles))
	Threads.@threads for (i,(key,profile)) ∈ collect(enumerate(profiles) )
		tel1 = key[1] 
		tel2 = key[2]
		key1 = "$tel1-$key" 
		key2 = "$tel2-$key" 

		wvlngth = get_wavelength(profile)

		thrs1 = median(spectra[key1].val) .* thrs
		thrs2 = median(spectra[key2].val) .* thrs


		λmin = wvlngth[min(findfirst(spectra[key1].val .> thrs1),findfirst(spectra[key2].val .> thrs2))]
		λmax = wvlngth[max(findlast(spectra[key1].val .> thrs1),findlast(spectra[key2].val .> thrs2))]
		@reset profile.λbnd = [λmin, λmax]
		pr_array[i] = key=>profile
	end
	return Dict(pr_array)

end


function gravi_compute_transmissions(  spectra::Dict{String, ConcreteWeightedData{T,N}},
										profiles::Dict{String,SpectrumModel{A,B,C}},
										lamp;
										thrs=0.01,
										kwds...) where {T,N,A,B,C<:Interpolator} 

	pr_array = Vector{Pair{String,SpectrumModel{A,B,C}}}(undef,length(profiles))
	Threads.@threads for (i,(key,profile)) ∈ collect(enumerate(profiles) )
		tel1 = key[1] 
		tel2 = key[2]
		key1 = "$tel1-$key" 
		key2 = "$tel2-$key" 
		BSp1 = profile.transmissions[1].basis
		BSp2 = profile.transmissions[2].basis


		wvlngth = get_wavelength(profile)
		lmp = lamp.(wvlngth)
		good = isfinite.(lmp) .&& (lmp .!= 0)
		wvgood = wvlngth[good]
		lmp = view(lmp,good)

		transmissions = [gravi_fit_transmission( view(spectra[key1],good),lmp,BSp1,wvgood; kwds...)
						gravi_fit_transmission( view(spectra[key2], good),lmp,BSp2, wvgood; kwds...)]
		@reset profile.transmissions = transmissions

		flat = ones(Float64,length(spectra[key1]))
		flat[good] .= ((view(spectra[key1],good) / (profile.transmissions[1].(wvgood).* lmp) + view(spectra[key2],good) / (profile.transmissions[2].(wvgood).* lmp))/2).val
		
		@reset profile.flat = flat
		pr_array[i] = key=>profile
	end
	profiles = Dict(pr_array)

	return profiles

end

function gravi_init_transmissions(profiles::Dict{String,SpectrumModel{A,B,C}};
									nb_transmission_knts=20,
									kernel = CatmullRomSpline(),
									kwds...
									) where {A,B,C} 
									
	λmin = minimum([get_wavelength(p,1) for p ∈ values(profiles)])
	λmax = maximum([get_wavelength(p,360) for p ∈ values(profiles)])
	knt =  range(λmin,λmax,nb_transmission_knts)



	pr_array = Vector{Pair{String,SpectrumModel{A,B,Interpolator{typeof(knt),typeof(kernel)}}}}(undef,length(profiles))
	for (i,(key,profile)) ∈ collect(enumerate(profiles) )
		λ = get_wavelength(profile)
		λ = λ[isfinite.(λ)]
		S = Interpolator(knt,kernel)
		initcoefs = compute_coefs(S,λ, ones(length(λ)))
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
									kwd...) where {T, A<:AbstractWeightedData{T, 1}} 
									
	coefs = compute_coefs(B,wavelength,spectrum/lampspectrum)
	return InterpolatedSpectrum(coefs,B)

end




function gravi_compute_lamp(spectra::Dict{String, D}, 
							profiles::Dict{String,SpectrumModel{A,B,C}};
							nb_lamp_knts=360,
							init_lamp = nothing,
							kernel = CatmullRomSpline(),
							kwds...) where {T,A,B,C<:Interpolator,D<:AbstractWeightedData{T, 1}} 
	
	data_trans = Vector{@NamedTuple{spectrum::WeightedData{Float64,1,SubArray{Float64, 1, Vector{Float64}, Tuple{Vector{Int64}}, false},SubArray{Float64, 1, Vector{Float64}, Tuple{Vector{Int64}}, false}},
									transmission::Vector{Float64},wavelength::B}}(undef,length(spectra))	
	for (i,(key,spectrum)) ∈ enumerate(spectra)		
		profile = profiles[ key[3:end]]
		transmission = key[1] == key[3] ? profile.transmissions[1] : profile.transmissions[2]
		wvlngth = get_wavelength(profile)
		good = .!isnan.(wvlngth)
		data_trans[i] = (;spectrum = view(spectrum,good), transmission = transmission(wvlngth[good]), wavelength = wvlngth[good])
	end
	local Bs

	if isnothing(init_lamp)
		λmin = minimum([get_wavelength(p,1) for p ∈ values(profiles)])
		λmax = maximum([get_wavelength(p,360) for p ∈ values(profiles)])
		knt = range(λmin,λmax,nb_lamp_knts)
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
	b = zeros(Float64,length(knots))
	A = zeros(Float64,length(knots),length(knots))
	Threads.@threads for (;spectrum,transmission, wavelength) ∈ data_trans
		K = build_interpolation_matrix(kernel,knots,wavelength) .* transmission
		b .+= K' * (spectrum.precision .* spectrum.val )
		A .+= K'* (spectrum.precision .* K)
	end
	return pinv(A) * b
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


 