function fitprofile(data::AbstractWeightedData{T,2},bndbx::C; center_degree=4, σ_degree=4, thrsld=0.1) where{T,C<:CartesianIndices}

	fulldata = view(data,bndbx)
	spectra = (sum(fulldata.val .* fulldata.precision,dims=2)./ sum(fulldata.precision,dims=2))[:]
	spectra[isnan.(spectra)].=T(0)
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
	center .*=  specmodel.preconditionner[1:center_degree+1]
	σ .*=  specmodel.preconditionner[1:σ_degree+1]
	θopt = (;center=center,σ=σ)
	return θopt
end

function gravi_extract_model(	data::AbstractWeightedData{T,N},
								profile::SpectrumModel; 
								restrict=0.01, 
								nonnegative=false,
								robust=false
								) where {T,N}
	bbox = profile.bbox
	if  ndims(bbox)<N
		(;val, precision) = view(data,bbox,:)
	else
		(;val, precision) = view(data,bbox)
	end
	model =  get_profile(profile)
	if restrict>0
		model .*= (model .> restrict)
	end

	αprecision = dropdims(sum(  model.^2 .* precision ,dims=2),dims=2)
	α = dropdims(sum(model .* precision .* val,dims=2),dims=2) ./ αprecision

	nanpix = .! isnan.(α)
	if nonnegative
		positive = nanpix .& (α .>= T(0))
	else
		positive = nanpix
	end

	if robust # Talwar hard descender
		res = sqrt.(precision) .* (positive .* α  .* model .- val) 
		
		good = (T(-2.795) .< res .<  T(2.795))
		αprecision =dropdims(sum( good .* model.^2 .* precision ,dims=2),dims=2)
		α = dropdims(sum(good .* model .* precision .* val,dims=2),dims=2) ./ αprecision
		
		nanpix = .! isnan.(α).*good
		if nonnegative
			positive = nanpix .& (α .>= T(0))
		else
			positive = nanpix
		end
	end

	return positive .* α .* model
end


function gravi_extract_profile(	data::AbstractWeightedData{T,N},
								profile::SpectrumModel; 
								restrict=0.01, 
								nonnegative=false, 
								robust=false,
								kwds...
								) where {T,N}
	bbox = profile.bbox
	if  ndims(bbox)<N
		(;val, precision) = view(data,bbox,:)
	else
		(;val, precision) = view(data,bbox)
	end
	model =  get_profile(profile)
	if restrict>0
		#model .*= (model .> restrict)
		precision .*=  (model .> restrict)
	end

	αprecision = sum(  model.^2 .* precision ,dims=2)
	α = sum(model .* precision .* val,dims=2) ./ αprecision

	nanpix = .! isnan.(α)
	if nonnegative
		positive = nanpix .& (α .>= T(0))
	else
		positive = nanpix
	end

	wd = WeightedData(dropdims(positive .* α,dims=2), dropdims(positive .* αprecision,dims=2))

	if robust # Talwar hard descender
		#@show size(precision), size(wd), size(model) , size(val)
		res = sqrt.(precision) .* (positive .* α  .* model .- val) 
		
		good = (T(-2.795) .< res .<  T(2.795))
		αprecision =dropdims(sum( good .* model.^2 .* precision ,dims=2),dims=2)
		α = dropdims(sum(good .* model .* precision .* val,dims=2),dims=2) ./ αprecision
		
		nanpix = .! isnan.(α)
		if nonnegative
			positive = nanpix .& (α .>= T(0))
		else
			positive = nanpix
		end
		wd = WeightedData(positive .* α, positive .* αprecision)
	end
	return wd / profile.flat
end

function gravi_extract_profile(	data::AbstractWeightedData{T,N},	
								profile::AbstractDict; 
								kwds...) where {T,N}
	profiles = Dict{String,ConcreteWeightedData{Float64,1}}()
	for (key,val) ∈ profile
		push!(profiles,key=>gravi_extract_profile(data ,val; kwds...))
	end
	return profiles
end

function gravi_extract_profile_flats(	flats::Vector{<:AbstractWeightedData{T,N}},
										profile::AbstractDict; 
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
								profile::AbstractDict; 
								kwds...) where {T,N}
	profiles = Dict{String,AbstractWeightedData{T,1}}()
	for (key,val) ∈ profile
		push!(profiles,key=>gravi_extract_profile(data,precision ,val; kwds...))
	end
	return profiles
end
