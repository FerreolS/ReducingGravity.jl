module ReducingGravity

using 	Accessors,
		ArrayTools, 
		ChainRulesCore,
		EasyFITS,
		FITSHeaders,
		InterpolationKernels,
		LinearAlgebra,
		Optim, 
		OptimPackNextGen,
		Optimisers, 
		PRIMA,
		SparseArrays,
		StaticArrays,
		Statistics, 
		StatsBase,
		Tullio,
		Zygote

import 	ImageFiltering: mapwindow


export 	gravi_data_create_bias_mask,
		gravi_data_detector_cleanup,
		gravi_compute_badpix,
		gravi_compute_blink,
		gravi_compute_profile,
		gravi_extract_profile,
		gravi_extract_profile_flats_from_p2vm,
		gravi_compute_gain_from_p2vm,
		gravi_create_weighteddata,
		gravi_spectral_calibration,
		gravi_compute_lamp_transmissions,
		gravi_build_p2vm_interf,
		gravi_build_V2PM,
		gravi_compute_ron,
		WeightedData,
		AbstractWeightedData,
		ConcreteWeightedData,
		combine,
		likelihood,
		flagbadpix!,
		listfitsfiles



include("utils/FITSutils.jl")
include("utils/utils.jl")
include("utils/WeightedData.jl")
include("utils/Interpolation.jl")
include("utils/DataStruct.jl")
include("gravi_profile.jl")
include("gravi_calib.jl")
include("gravi_wave.jl")
include("gravi_p2vm.jl")
include("gravi_recalib.jl")

#@enum Resolution LOW MED HIGH 


function gravi_data_create_bias_mask(darkfits::FitsFile)

	#read_key(darkfits[1],"ESO DPR TYPE")=="DARK" || error("must be dark file")
	xname = ["LEFT","HALFLEFT","CENTER","HALFRIGHT","RIGHT"]
	resolution = darkfits[1]["ESO INS SPEC RES"].string
	
	nx,ny,nf = darkfits["IMAGING_DATA_SC"].data_size
	illuminated = falses(nx,ny)
	IMAGING_DETECTOR_SC =read(darkfits["IMAGING_DETECTOR_SC"])

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
	bbx = Dict{String, Tuple{Vector{Float64},CartesianIndices{2, Tuple{UnitRange{Int64}, UnitRange{Int64}}}}}()
	sizehint!(bbx,nregion)
	rng = 1:nx
	for i ∈ 1:nregion
		y = Float64.([ IMAGING_DETECTOR_SC[xn][2,i] for xn ∈ xname])
		coefs = M*y
		cy = round.(Int, mapreduce( n -> coefs[n+1] .* rng.^n,+,0:maxdeg))
		for (ix,iy ) ∈ zip(rng,cy)
			view(illuminated,ix,max(1,iy-hw):min(ny,iy+hw)) .= true 
		end
		push!(bbx,IMAGING_DETECTOR_SC["REGNAME"][i]=>(coefs,CartesianIndices((1:nx,max(1,minimum(cy)-hw):min(ny,maximum(cy)+hw)))))
	end
	return (illuminated,bbx)		
end


function gravi_data_detector_cleanup!(	data::AbstractArray{T,3},
										illuminated::BitMatrix;
										keepbias=true,
										kwd...)  where T
	avgbias = zeros(T,size(data,1),1)
	
	@inbounds for n ∈ axes(illuminated,1)
		mdata = map(median, eachslice(data[n,.!illuminated[n,:],:],dims=2))
		view(data,n,:,:) .-= reshape(mdata,1,:)
		avgbias[n] = mean(mdata)
		
	end
	if keepbias
		data .+= avgbias
	end
	return  avgbias 
end



function gravi_data_detector_cleanup(data::AbstractArray{T,3},
										illuminated::BitMatrix;
										kwd...)  where T
	cleaneddata = copy(data)	
	avgbias = gravi_data_detector_cleanup!(cleaneddata,illuminated;kwd...)		
	return avgbias,cleaneddata			
end



function gravi_compute_badpix(	rawdata::AbstractArray{T,N},
								illuminated::BitMatrix; 
								spatialthresold=5, 
								spatialkernel=(5,5)) where {T,N}
	bias,data = gravi_data_detector_cleanup(rawdata,illuminated)
	if N==2
		mdata = data
	else
		mdata = map(x->quantile(x,0.75), eachslice(data,dims=(1,2)))

	end
	medfilt = mapwindow(median, mdata,spatialkernel,border="circular")
	sfiltered = (mdata .- medfilt) ./ max.(medfilt , bias)
	σs = mad(sfiltered)
	threshold = spatialthresold * σs
	return ( -threshold .< sfiltered .< threshold)

end

function gravi_compute_blink(	data::AbstractArray{T,3}; 
								goodpix= true(size(data)[1:2]),
								temporalthresold=5, 
								blinkkernel=5,
								bias::B=20,
								kwd...) where {T,B}


	ffiltered = similar(data)
	@inbounds @simd for i in findall(goodpix)
		mdata = quantile(data[i,:],0.5)
		b =  B <: Number ? bias :  bias[i[1]]
		
		ffiltered[i,:] = (data[i,:] .- mapwindow(median, data[i,:] ,blinkkernel,border="circular")) ./ sqrt.(max.(mdata ,b))
	end
	σ = mad(view(ffiltered,goodpix,:))
	threshold = T.(temporalthresold * σ)
	blink = ( -threshold .< ffiltered .< threshold)
	return blink
end

# unbiased estimator see  https://stats.stackexchange.com/questions/136976/find-an-unbiased-estimator-of-sigma-1  and "Aspects of multivariate statistical theory" 
function gravi_create_weighteddata(	data::AbstractArray{T,3},
									illuminated::BitMatrix,
									goodpix::BitMatrix; 
									cleanup = true,
									filterblink=true,
									unbiased=true, 
									bias=20,
									kwd...) where T
	
	sz =size(data)[1:2]	

	goodpix = copy(goodpix)
	if cleanup
		bias = gravi_data_detector_cleanup!(data,illuminated;kwd...)
	end

	if filterblink
		blink = gravi_compute_blink(data;bias=bias,goodpix=goodpix .&& illuminated,kwd...)
		goodpix .&= (sum(blink,dims=3) .> max(3,0.75 * size(blink,3)))
	else
		blink = trues(size(data))
	end
	
	avg = zeros(T,sz)
	wgt = zeros(T,sz)
	@inbounds @simd for i in CartesianIndices(sz)
		goodpix[i] &= illuminated[i]
		b = blink[i,:]
		sb = sum(b)
		d = data[i,:]
		view(blink,i,:) .&= goodpix[i]
		a = (sum(d.*b) ./ sb)
		if unbiased
			w =  (sum(b.*(d .- a).^2).\ (sb.- 3))
		else
			w =  (sum(b.*(d .- a).^2).\ (sb ))
		end
		goodpix[i] &= isfinite(w) && isfinite(a)
		if !goodpix[i] 
			a = T(0)
			w = T(0)
		end
		avg[i] = a
		wgt[i] = w
	end

	weighteddata = WeightedData(avg,wgt)
	flagbadpix!(weighteddata,.!goodpix)
	return (weighteddata, goodpix)
end


function gravi_create_weighteddata(	data::AbstractArray{T,3},
									illuminated::BitMatrix,
									goodpix::BitMatrix, 
									gain,
									ron;
									cleanup=true,
									keepbias=true,
									correct_gain=false,
									bndbox = CartesianIndices(goodpix),
									kwd...) where T
	

	if cleanup
		bias = gravi_data_detector_cleanup!(data,illuminated,keepbias=keepbias)
	end
	data = view(data ,bndbox,:)
	gain = view(gain,bndbox.indices[1])
	goodpix = view(goodpix,bndbox)
	avg = data.*goodpix
	wgt = goodpix ./ (ron .+ gain .\ max.(zero(T),avg) .+ T(1/12) )
	

	if correct_gain
		avg ./= gain
		wgt .*= gain.^2
	end	

	weighteddata = WeightedData(avg,wgt) 
	return weighteddata
end

function gravi_compute_ron(	dark::AbstractWeightedData,
	goodpix::BitMatrix, 
	gain) 
	return goodpix .* (1 ./dark.precision .- dark.val ./ gain)
end

function gravi_compute_profile(	flats::Vector{ConcreteWeightedData{T,N}},
								bboxes::Dict{String,Tuple{B,C}}; 
								center_degree=4, 
								σ_degree=4, 
								thrsld=0.1) where {T,N,C,B}
	profile = Dict{String,SpectrumModel{C,Nothing,Nothing,Nothing}}()
	Threads.@threads for tel1 ∈ 1:4
		for tel2 ∈ 1:4
			tel1==tel2 && continue
			flatsum = flats[tel1] + flats[tel2]
			for chnl ∈ ["A","B","C","D"]
				haskey(bboxes,"$tel1$tel2-$chnl-C") || continue
				bndbx =bboxes["$tel1$tel2-$chnl-C"][2] 
 				θ = fitprofile(flatsum,bndbx; center_degree=center_degree, σ_degree=σ_degree, thrsld=thrsld,center_guess=bboxes["$tel1$tel2-$chnl-C"][1]  )
				p = SpectrumModel(θ..., nothing, [0.,+Inf],Vector{InterpolatedSpectrum{Nothing,Nothing}}(),bndbx)
				push!(profile,"$tel1$tel2-$chnl-C"=>p) 
			end
		end
	end
	return profiles
end

function gravi_compute_profile(	flat::ConcreteWeightedData{T,N},
								bboxes::Dict{String,Tuple{B,C}}; 
								center_degree=4, 
								σ_degree=4, 
								thrsld=0.1) where {T,N,B,C}
	profiles = Dict{String,SpectrumModel{C,Nothing,Nothing,Nothing}}()
	Threads.@threads for (key, (center_guess,bndbx)) ∈ collect(bboxes)
		θ = fitprofile(flat,bndbx; center_degree=center_degree, σ_degree=σ_degree, thrsld=thrsld, center_guess=center_guess)
		p = SpectrumModel(θ..., nothing, [0.,+Inf],Vector{InterpolatedSpectrum{Nothing,Nothing}}(),bndbx)
		push!(profiles,key=>p) 
	end
	return profiles
end

function gravi_compute_lamp_transmissions(	spectra::Dict{String, ConcreteWeightedData{T,N}},
										profiles::Dict{String,SpectrumModel{A,B,C,T2}};
										lamp = nothing,
										loop=5,
										nb_transmission_knts=20,
										nb_lamp_knts=400,
										restart=false,		
										Chi2=  1.0,							
										kwds...) where {T,T2,N,A,B,C} 

	profiles,spectra = gravi_compute_wavelength_bounds(spectra,profiles)
	if C==Nothing || restart
		profiles = gravi_init_transmissions(profiles;T=T,	nb_transmission_knts=nb_transmission_knts,kwds...)
	elseif nb_transmission_knts != (length(first(values(profiles)).transmissions[1].coefs)) 
		profiles = gravi_init_transmissions(profiles;T=T,	nb_transmission_knts=nb_transmission_knts,kwds...)
	end

	
	for _ ∈ 1:loop
		lamp = gravi_compute_lamp(spectra,profiles; init_lamp=lamp, nb_lamp_knts=nb_lamp_knts, kwds...)
		profiles = gravi_compute_transmissions(spectra,profiles,lamp;Chi2=  Chi2, kwds...)
	end
	return (profiles, lamp)

end


function gravi_spectral_calibration(	wave::AbstractWeightedData{T, 2},
										darkwave::AbstractWeightedData{T, 2}, 
										profiles::Dict{String,SpectrumModel{A,B, C, E}}; 
										lines=argon[:,1],
										guess=argon[:,2],
										λorder=3,
										kwds...) where {A,B,C,T,E}

	wav = gravi_extract_profile(wave - darkwave, profiles;kwds...)
	new_profiles = Dict{String,SpectrumModel{A,Vector{Float64}, C,E}}()
	Threads.@threads for tel1 ∈ 1:4
		   for tel2 ∈ 1:4
				  tel1==tel2 && continue
				  for chnl ∈ ["A","B","C","D"]
						 chname = "$tel1$tel2-$chnl-C"
						 haskey(profiles,chname) || continue
						 updatedprofile = gravi_spectral_calibration(wav[chname] ,profiles[chname];lines=lines, guess=guess, λorder=λorder)
						 push!(new_profiles,chname=>updatedprofile) 
				  end
		   end
	end
	return new_profiles
end



end # module ReducingGravity