function fitprofile(data::AbstractWeightedData{T,2},
					bndbx::C; 
					center_degree=4, 
					σ_degree=4, 
					thrsld=0.1,
					center_guess=nothing,
					nb_σ=2,
					σ_guess=nothing) where{T,C<:CartesianIndices}

	fulldata = view(data,bndbx)
	spectra = (sum(fulldata.val .* fulldata.precision,dims=2)./ sum(fulldata.precision,dims=2))[:]
	spectra[isnan.(spectra)].=T(0)
	firstidx = findfirst(x -> x>mean(spectra)*thrsld,spectra)
	lastidx = findlast(x -> x>mean(spectra)*thrsld,spectra)


	data = view(data,bndbx[firstidx:lastidx,:])
	
	specmodel = ProfileModel(bndbx[firstidx:lastidx,:];maxdeg = max(center_degree,σ_degree), precond=true)



	center = zeros(center_degree+1)
	σ = zeros(σ_degree+1,nb_σ)
	if isnothing(center_guess)
		ax,ay =  bndbx[firstidx:lastidx,:].indices
		ax = Float64.(ax)
		c = reshape(sum(data.val.*sqrt.(data.precision).* ay', dims=2) ./ sum(sqrt.(data.precision).* data.val, dims=2),:)
		valid = isfinite.(c)
		Vandermonde = reduce(hcat,[ ax[valid].^n  for n ∈ 0:center_degree])
		B = diagm(1. ./sqrt.(sum(Vandermonde.^2,dims=1)[:]))
		VB = Vandermonde*B
		center = (B*((VB'*VB)\VB')*c[valid])[:]


	else
		l = min(length(center_guess),center_degree+1)
		center[1:l] = center_guess[1:l]
	end
	
	if isnothing(σ_guess)
		σ[1,:] .= 0.5 #std((shp .* ay) ./ sum(shp))
	else
		l = min(size(σ_guess,1),σ_degree+1)
		σ[1:l,:] .= σ_guess[1:l]
	end

	θ = (;center=center./specmodel.preconditionner[1:center_degree+1], σ = σ./specmodel.preconditionner[1:σ_degree+1])
	params, unflatten = destructure(θ)
	f(params) = scaledlikelihood(data,specmodel(;unflatten(params)...))
	xopt, info = prima(f, params; maxfun=10_000,ftarget=length(data))
	
	θopt= unflatten(xopt)
	(;center,σ) = θopt

	center .*=  specmodel.preconditionner[1:center_degree+1]
	σ .*=  specmodel.preconditionner[1:σ_degree+1]
	θopt = (;center=center,σ=σ)
	return θopt
end

#= A voir si on peut reestimer la precision avec le  model comme estimation de l'esperance
 =#


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


function gravi_extract_profile(data::AbstractWeightedData{T,N},
								profile::SpectrumModel; 
								restrict=0.01, 
								nonnegative=false, 
								robust=false,
								kwds...
								) where {T,N}
	bbox = CartesianIndices((get_wavelength_bounds_inpixels(profile),profile.bbox.indices[2]))
	if  ndims(bbox)<N
		(;val, precision) = view(data,bbox,:)
	else
		(;val, precision) = view(data,bbox)
	end
	model =  T.(get_profile(profile))
	if restrict>0
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
	return wd
end

function gravi_extract_profile(	data::AbstractWeightedData{T,N},	
								profile::AbstractDict; 
								kwds...) where {T,N}
	profiles = Dict{String,ConcreteWeightedData{T,1}}()
	for (key,val) ∈ profile
		push!(profiles,key=>gravi_extract_profile(data ,val; kwds...))
	end
	return profiles
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

function gravi_get_usefull_pixels(profiles::AbstractDict,
									goodpix::BitMatrix;
									restrict=0.0  )
	sz = size(goodpix)
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
	return ill .& goodpix 
end