
function build_wavelength_range2(profiles;
								padding=0, 
								λmin=0,
								λmax=1,
								narrow=false) 

	λstep = maximum([ (-((t=get_wavelength(p;bnd=true))[[end 1]]...))/(length(t)-1)  for p ∈ values(profiles)])

	bboxes = [get_wavelength_bounds_inpixels(profile) for (_, profile) ∈ profiles]
	minbox = minimum(first.(bboxes))
	maxbox = maximum(last.(bboxes))
	xindices = minbox:maxbox

	if narrow
		wvmin = max(λmin,maximum([p.λbnd[1] for p ∈ values(profiles)]))
		wvmax = min(λmax,minimum([p.λbnd[2] for p ∈ values(profiles)]))
	else
		wvmin = max(λmin,minimum([p.λbnd[1] for p ∈ values(profiles)]))
		wvmax = min(λmax,maximum([p.λbnd[2] for p ∈ values(profiles)]))
	end

	return range(wvmin - padding * λstep,wvmax  +padding *  λstep; step = λstep),xindices

end


function gravi_extract_profile(data::AbstractWeightedData{T,N},
								profile::SpectrumModel,
								bbox::CartesianIndices; 
								restrict=0.01, 
								nonnegative=false, 
								robust=0,
								kwds...
								) where {T,N}
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

	if robust>0 # Talwar hard descender
		res = sqrt.(precision) .* (positive .* α  .* model .- val) 
		
		good = (T(-2.795*robust) .< res .<  T(2.795*robust))
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


function gravi_extract_profile_array(	data::AbstractWeightedData{T,N},	
								profiles::AbstractDict,
								xindices::UnitRange; 
								kwds...) where {T,N}
	nchannels = length(profiles)
	boxlen = length(xindices)

	if N == 2
		pval = zeros(T,boxlen,nchannels)
		pprec = zeros(T,boxlen,nchannels)
	else
		pval = zeros(T,boxlen,nchannels, size(data)[3:end]...)
		pprec = zeros(T,boxlen,nchannels, size(data)[3:end]...)
	end
	sortedprofiles = sort!(collect(profiles), by= x->last(x).center[1])
	Threads.@threads for (i, (_, profile)) ∈ collect(enumerate(sortedprofiles))
		bbox = CartesianIndices((xindices,profile.bbox.indices[2]))
		extracted = gravi_extract_profile(data,profile,bbox; kwds...)
		pval[:,i,..] .= extracted.val[..]
		pprec[:,i,..] .= extracted.precision[..]
	end
	return WeightedData(pval,pprec)
end

function gravi_lamp_estimation(data::AbstractWeightedData{T,N},
								flat,
								transmission,
								interpolation_matrices,
								integration_matrices) where {T,N}
							
	return lamp
end

function make_convolution_integration_matrices(profiles,
									knots::AbstractRange{T},
									pixindices::AbstractVector;
									kernel= CatmullRomSpline{T}()) where T
									
	nchannels = length(profiles)
	CI_matrices = Vector{SparseMatrixCSC{T,Int}}(undef,nchannels)
	sortedprofiles = sort!(collect(profiles), by= x->last(x).center[1])

	K = build_sparse_interpolation_integration_matrix(kernel,knots, map(T,pixindices))
	Threads.@threads for (i, (_, profile)) ∈ collect(enumerate(sortedprofiles))
		lsf = get_LSF(profile,knots)
		C = build_convolution_matrix(lsf)
		CI_matrices[i] = K*C 
	end


	return CI_matrices
	
end


function get_LSF_size((;bbox,center,σ,λ)::SpectrumModel,ax)

	ncenter = length(center)
	nσ = size(σ,1)
	ay = bbox.indices[2]

	degmax = max(ncenter,nσ)
	
	u = broadcast(^,Float64.(ax),(0:(degmax-1))')
		
	cy = u[:,1:ncenter]*center
	
	sy = u[:,1:nσ]*σ
	left = abs.(ay[end] .- cy)
	right = abs.(ay[1] .- cy)
	return sum(hcat(left ,right ).*sy,dims=2)./(left.+right)
end

function get_LSF(profile,ax;hwdth=2,upsamplefactor=1) 
	s = get_LSF_size(profile,ax) .* upsamplefactor
	#s .= mean(s)
	S = exp.(-1 ./ 2 .*(((-hwdth:hwdth))' ./ s).^2)
	S = S ./ sum(S,dims=2)
	return OffsetArray(S,:,-hwdth:hwdth)
end



function build_convolution_matrix(lsf; T=Float32) 
	npix = size(lsf,1)
	

	nC = length(lsf)
	CL = zeros(T,nC)
	CC = zeros(T,nC)
	CV = zeros(T,nC)
	minl = minimum(axes(lsf,2))
	maxl = maximum(axes(lsf,2))

	iC = 1 
 	for l ∈ axes(lsf,1)	
		iCi = max(l + minl,1) 
		iCe = min(l + maxl,npix)
		CL[iC:(iC+iCe-iCi)] .= iCi:iCe
		CC[iC:(iC+iCe-iCi)] .= l
		CV[iC:(iC+iCe-iCi)] .= lsf[l,(iCi-l):(iCe-l)]
		iC = iC+iCe-iCi+1

		#K[l,off:(off+wsz-1)] .= weights
	end
	C = sparse(CL[1:iC-1],CC[1:iC-1],CV[1:iC-1],npix,npix)
	return C
end

function build_spectral_interpolation_matrices(profiles,
												λknots::AbstractRange{T},
												pixwavelength::AbstractVector;
												kernel= CatmullRomSpline{T}()) where T
	nchannels = length(profiles)
	S_matrices = Vector{SparseMatrixCSC{T,Int}}(undef,nchannels)
	sortedprofiles = sort!(collect(profiles), by= x->last(x).center[1])

	Threads.@threads for (i, (_, profile)) ∈ collect(enumerate(sortedprofiles))
		S_matrices[i] = build_sparse_interpolation_matrix(kernel,λknots, map(T,pixwavelength))
	end


	return S_matrices

end


function build_interpolation_convolution_integration_matrices(profiles,
									knots::AbstractRange,
									kernel::Kernel{T,N},
									pixindices::UnitRange;
									upsamplefactor::Int=2)  where{T,N}
	nchannels = length(profiles)
	ICI_matrices = Vector{SparseMatrixCSC{T,Int}}(undef,nchannels)
	sortedprofiles = sort!(collect(profiles), by= x->last(x).center[1])
	knots = upsample(knots,upsamplefactor)
	pixknots = upsample(pixindices,upsamplefactor)
	K = build_sparse_interpolation_integration_matrix(kernel,pixknots, map(T,pixindices))
	Threads.@threads for (i, (_, profile)) ∈ collect(enumerate(sortedprofiles))
		lsf = get_LSF(profile,pixknots;upsamplefactor=upsamplefactor)
		C = build_convolution_matrix(lsf)
		λ  = get_wavelength(profile)[pixindices]
		pixwavelength = upsample(λ,upsamplefactor)
		I = build_sparse_interpolation_matrix(kernel,knots, map(T,pixwavelength))
		ICI_matrices[i] = K*C*I 
	end


	return ICI_matrices
	
end

function upsample(v::AbstractVector{T},factor::Int) where T
	n = length(v)
	out = similar(v,factor*(n-1)+1)
	@inbounds for i ∈ 1:n-1
		out[(i-1)*factor+1:i*factor] .= range(v[i],v[i+1],factor+1)[1:end-1]
	end
	out[end] = v[end]
	return out
end

function upsample(v::AbstractRange,factor::Int) 
	return range(first(v),last(v),step=(step(v)/factor))
end


function f!(y, ICI_matrices,s)
	#AK.foraxes(y, 1) do i
	#	y[i,:] .= ICI_matrices[i]*s
	#end


end


function gravi_extract_profile_flats_from_p2vm(	flats::Vector{W}, 
	chnames::Matrix{String} ,
	profiles::AbstractDict,
	xindices::UnitRange;
	kwds...
	) where {T,W<:AbstractWeightedData{T, 2}} 
	
	
	nchannels = length(profiles)
	boxlen = length(xindices)

	

	pval = zeros(T,boxlen,nchannels,2)
	pprec = zeros(T,boxlen,nchannels,2)
	
	sortedprofiles = sort!(collect(profiles), by= x->last(x).center[1])
	Threads.@threads for (i, (key, profile)) ∈ collect(enumerate(sortedprofiles))
		for j∈1:2
			chnl = "$(key[j])-$key"		
			idx = [idxt[1] for idxt ∈ findall( x -> x == chnl,  chnames)]
			bbox = CartesianIndices((xindices,profile.bbox.indices[2]))
			idx = [idxt[1] for idxt ∈ findall( x -> x == chnl,  chnames)]
			ch1 = gravi_extract_profile(flats[idx[1]] ,profile,bbox;kwds...)
			ch2 = gravi_extract_profile(flats[idx[2]] ,profile,bbox;kwds...)
			extracted = combine(ch1,ch2)
			pval[:,i,j] .= extracted.val[..]
			pprec[:,i,j] .= extracted.precision[..]
		end
	end
	return WeightedData(pval,pprec)
end

function init_lamp_transmission(flats::AbstractWeightedData{T, 3},profiles;nb_transmission_knts=20) where T
	nchannels = length(profiles)
	transmissions = Matrix{T}(undef,nchannels,2)
end