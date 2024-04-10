module ReducingGravity

using 	FITSIO,
		LinearAlgebra,
		Statistics, 
		ArrayTools, 
		ImageFiltering,
		StatsBase,
		Statistics,
		ConcreteStructs,
		Optim, 
		Optimisers, 
		StaticArrays,
		BSplineKit,
		Zygote,
		ChainRulesCore,
		ParameterHandling,
		Accessors,
		OptimPackNextGen

export 	gravi_data_create_bias_mask,
		gravi_data_detector_cleanup,
		gravi_compute_badpix,
		gravi_compute_blink,
		gravi_compute_profile,
		gravi_extract_profile,
		gravi_extract_profile_flats,
		gravi_create_weighteddata,
		gravi_spectral_calibration
		WeightedData,
		likelihood



include("utils/FITSutils.jl")
include("utils/DataStruct.jl")
include("gravi_calib.jl")
include("gravi_wave.jl")

#@enum Resolution LOW MED HIGH 


function gravi_data_create_bias_mask(darkfits::FITS)

	#read_key(darkfits[1],"ESO DPR TYPE")=="DARK" || error("must be dark file")
	xname = ["LEFT","HALFLEFT","CENTER","HALFRIGHT","RIGHT"]
	resolution = read_key(darkfits[1],"ESO INS SPEC RES")[1]
	
	nx,ny,nf = size(darkfits["IMAGING_DATA_SC"])
	illuminated = falses(nx,ny)
	IMAGING_DETECTOR_SC = Dict(darkfits["IMAGING_DETECTOR_SC"])

	nregion = length(IMAGING_DETECTOR_SC["REGNAME"])

	
	if resolution=="HIGH"
		maxdeg = 2
		hw = 6
	else 
		maxdeg = 0
		hw = 4
	end
	
	x = Float64.([ IMAGING_DETECTOR_SC[xn][1] for xn ∈ xname])

	Vandermonde = reduce(hcat,[ x.^n  for n ∈ 0:maxdeg])
	M = (Vandermonde'*Vandermonde)\Vandermonde'
	# Preconditionning
	# B = diagm(1. ./sqrt.(sum(Vandermonde.^2,dims=1)[:]))
	# C = Vandermonde*diagm(b[:])
	# M = B*((C'*C)\C')
	
	#bbx = Dict{String,BoundingBox{Int}}()
	bbx = Dict{String, CartesianIndices{2, Tuple{UnitRange{Int64}, UnitRange{Int64}}}}()
	sizehint!(bbx,nregion)
	rng = 1:nx
	for i ∈ 1:nregion
		y = Float64.([ IMAGING_DETECTOR_SC[xn][2,i] for xn ∈ xname])
		coefs = M*y
		cy = round.(Int, mapreduce( n -> coefs[n+1] .* rng.^n,+,0:maxdeg))
		for (ix,iy ) ∈ zip(rng,cy)
			illuminated[ix,max(1,iy-hw):min(ny,iy+hw)] .= true 
		end
		#push!(bbx,IMAGING_DETECTOR_SC["REGNAME"][i]=>BoundingBox(1,nx,max(1,minimum(cy)-hw),min(ny,maximum(cy)+hw)))
		push!(bbx,IMAGING_DETECTOR_SC["REGNAME"][i]=>CartesianIndices((1:nx,max(1,minimum(cy)-hw):min(ny,maximum(cy)+hw))))
	end
	return (illuminated,bbx)		
end


function gravi_data_detector_cleanup(	data::AbstractArray{T,3},
										illuminated::BitMatrix)  where T
	avgbias = T(0)
	cleaneddata = copy(data)
	@inbounds for n ∈ axes(illuminated,1)
		#mdata = median(data[n,.!illuminated[n,:],..],dims=1) 
		mdata = map(median, eachslice(data[n,.!illuminated[n,:],:],dims=2))
		cleaneddata[n,:,..] .-= reshape(mdata,1,:)
		avgbias += mean(mdata)
	end
	avgbias /= size(illuminated,1)
	return  avgbias,cleaneddata
end



function gravi_compute_badpix(	rawdata::AbstractArray{T,N},
								illuminated::BitMatrix; 
								spatialthresold=5, 
								spatialkernel=(5,5)) where {T,N}
	bias,data = gravi_data_detector_cleanup(rawdata,illuminated)
	if N==2
		mdata = data
	else
		#mdata = dropdims(median(data, dims=3),dims=3)
		#mdata = dropdims(mapslices(x->quantile((x),0.75), data, dims=3),dims=3) # to account for higly blinking pixels  # median(, dims=) is type unstable
		mdata = map(x->quantile(x,0.75), eachslice(data,dims=(1,2)))

	end
	medfilt = mapwindow(median, mdata,spatialkernel,border="circular")
	sfiltered = (mdata .- medfilt) ./ max.(medfilt , bias)
	σs = mad(sfiltered)
	threshold = spatialthresold * σs
	return ( -threshold .< sfiltered .< threshold)

end

function gravi_compute_blink(	data::AbstractArray{T,3}; 
								temporalthresold=5, 
								blinkkernel=(1,1,5),
								bias=20,
								kwd...) where {T}
	bias = T(bias)
	#mdata = map(median, eachslice(data,dims=(1,2)))
	#mdata = dropdims(median(data, dims=3),dims=3) # median(, dims=) is type unstable
	mdata = map(x->quantile(x,0.5), eachslice(data,dims=(1,2)))
	ffiltered = (data .- mapwindow(median, data,blinkkernel,border="circular")) ./ max.(mdata , bias)

	σ = mad(ffiltered)
	threshold = T.(temporalthresold * σ)
	blink = ( -threshold .< ffiltered .< threshold)
	return blink
end

# unbiased estimator see  https://stats.stackexchange.com/questions/136976/find-an-unbiased-estimator-of-sigma-1  and "Aspects of multivariate statistical theory" 
function gravi_create_weighteddata(	rawdata::AbstractArray{T,3},
									illuminated::BitMatrix,
									goodpix::BitMatrix; 
									unbiased=true, 
									kwd...) where T
	
	goodpix = copy(goodpix)

	bias,data = gravi_data_detector_cleanup(rawdata,illuminated)

	if filterblink
		blink = gravi_compute_blink(data,bias=bias;kwd...)
		goodpix .&= (sum(blink,dims=3) .> max(3,0.75 * size(blink,3)))
	else
		blink = trues(size(data))
	end
	avg = (sum(data.*goodpix.*blink, dims=3) ./ sum(goodpix.*blink, dims=3))[:,:,1]
	if unbiased
		wgt =  (sum(goodpix.*blink.*(data .- avg).^2,dims=3).\ (sum(goodpix.*blink,dims=3) .- 3))[:,:,1]
	else
		wgt =  (sum(goodpix.*blink.*(data .- avg).^2,dims=3).\ (sum(goodpix.*blink,dims=3) ))[:,:,1]
	end
	avg[.!(goodpix)] .= zero(T)
	wgt[.!(goodpix)] .= zero(T)
	weighteddata = WeightedData(avg,wgt)
	flagbadpix!(weighteddata,.!goodpix)
	return (weighteddata, goodpix)
end

function gravi_compute_profile(	flats::Vector{<:AbstractWeightedData{T,N}},
								bboxes::Dict{String,C}; 
								center_degree=4, 
								σ_degree=4, 
								thrsld=0.1) where {T,N,C}
	profile = Dict{String,Profile{C}}()
	Threads.@threads for tel1 ∈ 1:4
		for tel2 ∈ 1:4
			tel1==tel2 && continue
			flatsum = flats[tel1] + flats[tel2]
			for chnl ∈ ["A","B","C","D"]
				haskey(bboxes,"$tel1$tel2-$chnl-C") || continue
				bndbx =bboxes["$tel1$tel2-$chnl-C"] 
				(;val, precision) = view(flatsum,bndbx)
 				θ = fitprofile(val, precision,bndbx; center_degree=center_degree, σ_degree=σ_degree, thrsld=thrsld )
				p = Profile(θ...,Vector{Float64}(),bndbx)
				push!(profile,"$tel1$tel2-$chnl-C"=>p) 
			end
		end
	end
	return profile
end

function gravi_extract_profile(	data::AbstractWeightedData{T,N},
								profile::Profile; 
								restrict=0.01, 
								nonnegative=false, 
								robust=false) where {T,N}
	bbox = profile.bbox
	(;val, precision) = view(data,bbox)

	model =  SpectrumModel{T}(bbox)(profile)
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
								profile::Dict{String,<:Profile}; 
								kwds...) where {T,N}
	profiles = Dict{String,AbstractWeightedData{T,1}}()
	for (key,val) ∈ profile
		push!(profiles,key=>gravi_extract_profile(data ,val; kwds...))
	end
	return profiles
end

function gravi_extract_profile_flats(	flats::Vector{<:AbstractWeightedData{T,N}},
										profile::Dict{String,<:Profile}; 
										kwds...) where {T,N}
	spctr = Dict{String,AbstractWeightedData{T,1}}()
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


function gravi_compute_transmission(	spectra::Dict{String, AbstractWeightedData{T, 1}};  
										verb=10,maxeval=50, kwd...) where T 
	spectraArray =values(spectra)
	nspectra = length(spectra)
	meanspectrum = sum(spectraArray) / nspectra
	rng= 1:length(meanspectrum)
	knt = SVector{18,Float32}(1.0, 24.0, 35.0, 41.0, 46.0, 58.0, 69.0, 91.0, 114.0, 125.0, 136.0, 159.0, 181.0, 226.0, 271.0, 294.0, 316.0, 360.0)
	#sp4 = Spline1D(1:360, meanspectrum.val; w=meanspectrum.precision, k=3, bc="zero",s=0.01)
	B = BSplineBasis(BSplineOrder(3), knt)
	ncoefs = length(B)
	coefs = [ [zeros(T,3)...,ones(T,ncoefs-6)...,zeros(T,3)...] for i ∈ 1:nspectra]
	#coefs = [ones(T,ncoefs) for i ∈ 1:nspectra]
	lamp = meanspectrum.val

	
	x = (;coefs=coefs)
	x0, restructure = destructure(x)
	function loss(;coefs= coefs, lamp=lamp) 
		S = Spline.(B,coefs)
		return mapreduce((x,y)->likelihood(y,map(x,rng) .* lamp),+,S,spectraArray)
	end
	f(x) = loss(;restructure(x)...)
	xopt = vmlmb(f,  x0 ;autodiff=true, verb=verb,maxeval=maxeval,kwd...)
	(;coefs) = restructure(xopt)

	transmissions = Dict(val=>Transmission(coefs[i],B) for (i,val) ∈ enumerate(keys(spectra)))
	return (transmissions, lamp)

end


function gravi_spectral_calibration(	wave::AbstractWeightedData{T, 2},
										darkwave::AbstractWeightedData{T, 2}, 
										profile::Dict{String,<:Profile}; 
										lines=argon[:,1],
										hw=2,
										λorder=3) where T

	wav = gravi_extract_profile(wave - darkwave, profile)
	Threads.@threads for tel1 ∈ 1:4
		   for tel2 ∈ 1:4
				  tel1==tel2 && continue
				  for chnl ∈ ["A","B","C","D"]
						 chname = "$tel1$tel2-$chnl-C"
						 haskey(profile,chname) || continue
						 updatedprofile = gravi_spectral_calibration(wav[chname] ,profile[chname];hw=hw, lines=lines,λorder=λorder)
						 @show updatedprofile
						 push!(profile,chname=>updatedprofile) 
				  end
		   end
	end
	return profile
end
end # module ReducingGravity
