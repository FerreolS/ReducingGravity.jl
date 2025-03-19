struct DispModel
    λ0::Float64
    coefs::Vector{Float64}
end

function gravi_extract_disp_model(disp_filename::String)
	wave0 = readfits(Vector{Float64}, disp_filename,"WAVE0",ext="DISP_MODEL")
	nmean = readfits(Matrix{Float64}, disp_filename,"N_MEAN",ext="DISP_MODEL")
	return [ DispModel(wave0[i],nmean[:,i]) for i in 1:length(wave0) ]
end

function get_refractive_index(disp::DispModel,k)
	return disp.coefs[1]  .+ sum([c .* (k .* disp.λ0 .- 1).^(i) for (i,c) ∈ enumerate(disp.coefs[2:end])	]) 
end

function groupdelay2OPL(gd::Vector{Vector{T}}) where T <:AbstractFloat
	# M = @SMatrix [ 1 -1 0 0
    #    1 0 -1 0
    #    -1 0 0 1
    #    0 1 -1 0
    #    0 -1 0 1
    #    0 0 -1 1]
	# ML = pinv(M'M)*M'
	ML = @SMatrix [ 0.25	0.25	-0.25	0.0		0.0		0.0
					-0.25   0.0  	0.0	   	0.25    -0.25	0.0
					0  		-0.25	0.0		-0.25	0		-0.25
					0		0		0.25    0	   	0.25    0.25]
	return hcat(ML*gd...)'
end

function recompute_phasors(data, k, opd)
	nλ = length(k)
	nt = size(opd,2)
	envlp = gravi_compute_envelope(opd,1 ./ k);
	ϕ = reshape(2π .*k .* opd , 1,nλ, nt)
	A = cat(cos.(ϕ),-sin.(ϕ), dims=1).*reshape(envlp,1,nλ, nt)
	c = zeros(ComplexF64,nλ)
	for l ∈ 1:nλ
		H = A[:,l,:]'
		c[l] = complex((pinv(H'*(data.precision[l,:].*H))*H' *(data.precision[l,:].*data.val[l,:]))...)
	end
	return c 
end



function recalibrate_wavenumber(data, k,	phasors, opd ; degmax=4,maxeval=500)
	ax = axes(data,1)
	preconditionner  = [ sqrt(length(ax) /sum(Float64.(ax).^(2*n)) ) for n ∈ 0:degmax]
	L = broadcast(^,Float64.(ax),(0:(degmax-1))').* preconditionner[1:degmax]'
	envlp = gravi_compute_envelope(opd,1 ./ k)
	ck_init = 	inv(L'*L)*L'*k
	F = envlp .* phasors
	function opt_wavenumber(ck)
		likelihood(data, real.(F.* exp.(1im .* 2π .* (L*ck) .* opd )))
	end
	return L * vmlmb(opt_wavenumber,  ck_init ;autodiff=true, maxeval=maxeval)
end

function recalibrate(data,visdata,dispmodel,fλ, profiles::AbstractDict; baselines=baselines_list, iter=1)

	kt = [ [1. ./ get_wavelength(profiles["$(baseline[1])$(baseline[2])-$chnl-C"];bnd=true)  for (k,chnl) ∈ enumerate(["A","B","C","D"])] for (i,baseline) ∈ enumerate(baselines)]
	
	_, slopes =  afine_model(visdata, fλ; lmin=20,lmax=200)
	opl = groupdelay2OPL(slopes)
	
	phasorst = [ Vector{Vector{ComplexF64}}(undef,4) for _ ∈ 1:6]
	for _ ∈ 1:iter
		kt,phasorst = recalibrate(data,opl,dispmodel, kt, phasorst; baselines=baselines)
		opl = estimate_opl!(data,dispmodel, kt,phasorst,opl)
	end
	return kt,phasorst,opl
end

function recalibrate(data,opl,dispmodel, kt,phasorst; baselines=baselines_list)
	Threads.@threads for (i,baseline) ∈ collect(enumerate(baselines))
		T1,T2 = baseline
		Threads.@threads for (j,chnl) ∈ collect(enumerate(["A","B","C","D"]))
			k = kt[i][j] 
			n1 = get_refractive_index(dispmodel[T1],k)
			n2 = get_refractive_index(dispmodel[T2],k)
		#	n1[:] .= 1.0
		#	n2[:] .= 1.0
			opd  = n1.*opl[T1,:]' .- n2 .* opl[T2,:]'
			d = view(data,j,i,:,:)
			phasorst[i][j] = recompute_phasors(d, k, opd)
			kt[i][j] = recalibrate_wavenumber(d, k, phasorst[i][j], opd)
		end
	end
	return kt,phasorst
end

function normalize_data(data, S, photometry; baselines=baselines_list)
	nl,nt = size(photometry[1])
	normalized = deepcopy(WeightedData(reshape(data.val,4,6,:,nt), reshape(data.precision,4,6,:,nt)))
	nλ = size(normalized,3)
	SS = reshape(S,4,6,nλ,16,nl)


	for (i,baseline) ∈ collect(enumerate(baselines))
		T1,T2 = baseline
		for (j,chnl) ∈ collect(enumerate(["A","B","C","D"]))
			d = normalized[j,i,:,:]
			photo1 = max.(0,SS[j,i,:,T1,:]*photometry[T1])
			photo2 = max.(0,SS[j,i,:,T2,:]*photometry[T2])
			denom = sqrt.( photo1 .* photo2)
			nd = (d - photo1 - photo2) / 2 / denom
			nd.val[denom.==0] .= 0
			nd.precision[denom.==0] .= 0
			view(normalized.val,j,i,:,:) .= nd.val
			view(normalized.precision,j,i,:,:) .= nd.precision
		end
	end
	return normalized
end


function afine_model(A::AbstractArray{Complex{T},2}, λ; lmin=1,lmax=size(A,1)) where T<:AbstractFloat
	ϕ = angle.(A)
	unwrap!(ϕ,dims=2)
	N = size(ϕ,2)
	size(ϕ,1) == length(λ) || throw(DimensionMismatch("The number of lines of ϕ must be equal to the length of λ"))
	w = T(2π) ./λ[lmin:lmax]
	#w0 =  T(2π) ./λ[(lmin+lmax)÷2]
	#w .-= w0
	intercept = Vector{T}(undef,N)
	slope = Vector{T}(undef,N)
	@inbounds @simd for n ∈ 1:N
		intercept[n], slope[n] =  affine_solve(ϕ[lmin:lmax,n],w)
	end
	return intercept, slope 
end

function afine_model(visdata::Vector{Matrix{Complex{T}}}, λ::AbstractVector{<:AbstractFloat}; lmin=1,lmax=size(visdata[1],1)) where {T}
	N = length(visdata)
	slope = Vector{Vector{T}}(undef, N)
	intercept = Vector{Vector{T}}(undef, N)
	for t ∈ 1:N
		intercept[t], slope[t] =  afine_model(visdata[t][lmin:lmax,:], λ[lmin:lmax])
	end
	return intercept, slope 
end

function afine_model(visdata::Vector{Matrix{Complex{T}}}, λ::AbstractVector{<:AbstractVector}; lmin=1,lmax=size(visdata[1],1)) where {T}
	N = length(visdata)
	slope = Vector{Vector{T}}(undef, N)
	intercept = Vector{Vector{T}}(undef, N)
	for t ∈ 1:N
		intercept[t], slope[t] =  afine_model(visdata[t][lmin:lmax,:], λ[t][lmin:lmax])
	end
	return intercept, slope 
end

function rescale_wavenumber(slopes::Vector{Vector{T}}) where {T}
	M = @SMatrix [	1 	-1 	0 	1 	0 	0
					1 	0 	1 	0 	-1 	0
					0 	1 	1 	0 	0 	-1
					0 	0 	0 	1 	1 	-1]
	t = length(slopes[1])
	B = @MArray zeros(Float64,t*4,6);
	@tullio B[(k-1)*512 + g,j] = M[k,j]*slopes[j][g]
	B1 = @view B[:,1]
	B2 = @view B[:,2:6]
	scales = vcat(1,.-(inv(B2'*B2)*B2'*(B1)))
	return scales
end

function recompute_wavelegnth(visdata::Vector{Matrix{Complex{T}}}, λ; lmin=1,lmax=size(visdata[1],1)) where {T}	
	N = length(visdata)
	intercepts, slopes = afine_model(visdata, λ; lmin=lmin,lmax=lmax)
	scw = rescale_wavenumber(slopes)
	finalλ = Vector{Vector{Float64}}(undef, N)
	for t ∈ 1:N
		intercepts[t], slopes[t] =  afine_model(visdata[t][lmin:lmax,:], λ[lmin:lmax] ./ scw[t])

		unwrp = unwrap(intercepts[t].%(2π))
		b,a = affine_solve(unwrp,slopes[t])
		w  = 2π ./  λ .* scw[t]
		finalλ[t] = 2π ./ (w .+ a)
	end
	meansc = mean(sum.((.*).([λ],finalλ)) ./ sum.((.*).(finalλ,finalλ)))
	return meansc .* finalλ
end

function reshape_pipeline_data(oidata::Matrix{Complex{T}}; baselines=baselines_list) where T
	output = Vector{Matrix{Complex{T}}}(undef,length(baselines))
	for (i,baseline) ∈ collect(enumerate(baselines))
		T1,T2 = baseline
		if T1>T2
			output[i] = deepcopy(oidata[:,i:6:end] )
		else
			output[i] = deepcopy(conj.(oidata[:,i:6:end]) )
		end
	end
	return output
end


function build_BMatrix(dispmodel,kt,phasors;baselines=baselines_list)
	nk = length(kt[1][1])
	B = zeros(Float64,6*4*nk,4)
	P = ones(ComplexF64,6*4*nk)
	for (b,baseline) ∈ collect(enumerate(baselines))
		T1,T2 = baseline	
		for c ∈ 1:4
			n1 = get_refractive_index(dispmodel[T1],kt[b][c])
			n2 = get_refractive_index(dispmodel[T2],kt[b][c])	
			for k ∈ 1:nk
				B[24*(k-1) + 4*(b-1)+c,T1] = n1[k].*kt[b][c][k].*2π
				B[24*(k-1) + 4*(b-1)+c,T2] = -1 * n2[k] * kt[b][c][k] *2π
				P[24*(k-1) + 4*(b-1)+c] = phasors[b][c][k]			
		   end
	   end
	end
	return B, P
end

function estimate_opl!(data,dispmodel, k,phasors,opl)
	nk = length(k[1][1])
	B,P = build_BMatrix(dispmodel,k,phasors)
	ndata = WeightedData(reshape(data.val,4*6*nk,:),reshape(data.precision,4*6*nk,:))
	nt = size(ndata,2)
	
	function opt_opl(t,_opl)
		likelihood(ndata[:,t], real.(P.* exp.(1im .* B*vcat(opl[1,t], _opl))))
	end
	Threads.@threads for t ∈ 1:nt
		opl[2:4,t] .= vmlmb(Base.Fix1(opt_opl,t), opl[2:4,t] ;autodiff=true, maxeval=500,xtol = (0.0,1e-9),ftol = (0.0,1e-18), gtol = (0.0,1e-16))
	end
	return opl
end


function gravi_recalibrate_wavelength(profiles::Dict{String,SpectrumModel{A,B, C, E}},
									newk::Vector{Vector{Vector{Float64}}}; 
									baselines=baselines_list
                                          ) where {A,B,C,E}

	new_profiles = Dict{String,SpectrumModel{A,Vector{Float64}, C,E}}()

    nλ = size(first(values(profiles)).bbox,1)
	for (b,baseline) ∈ collect(enumerate(baselines))
		T1,T2 = baseline
		for (c,chnl) ∈ collect(enumerate(["A","B","C","D"]))
			name = "$T1$T2-$chnl-C"
			wv = get_wavelength(profiles[name])
			newl = 1 ./ newk[b][c][:]
			λmin = minimum(newl)
	  		λmax = maximum(newl)
			wv[.!(isnan.(wv))] .= newl
			profile = profiles[name]
			@reset profile.λbnd = [λmin, λmax]
			@reset profile.λ = wv
			push!(new_profiles,name=>profile)
		end
	end
	return new_profiles
end




function gravi_build_V2PM(profiles::AbstractDict,
						baseline_phasors::Vector{Array{T, 3}},
						recalibrated_phasors::Vector{Vector{Vector{ComplexF64}}};
						baselines::Vector{Vector{Int64}}=baselines_list,
						λsampling=nothing,
						λmin=0.0,λmax=1.0,
						kernel = CatmullRomSpline{Float64}()) where {T}
	
	lk = length(kernel) 
	if isnothing(λsampling)
		λsampling =  build_wavelength_range(profiles;  padding=lk,λmin=λmin,λmax=λmax)
	end		
	λmin = max(minimum(λsampling),λmin)
	λmax = min(maximum(λsampling),λmax)
	usable_wvlngth = get_selected_wavelenght(profiles,baselines=baselines,λmin=λmin,λmax=λmax)
	gravi_build_V2PM(profiles, baseline_phasors, recalibrated_phasors, λsampling,usable_wvlngth,baselines,kernel)
end


function gravi_build_V2PM(	profiles::AbstractDict,
							baseline_phasors::Vector{Array{T, 3}},
							recalibrated_phasors::Vector{Vector{Vector{ComplexF64}}},
							λsampling,
							usable_wvlngth,
							baselines,
							kernel) where {T}
	
	lk = length(kernel) 
	nλ = length(λsampling)

	
	nmeasuredλ = maximum(maximum(diff(w,dims=1)) for w ∈ usable_wvlngth) +1				

	nL = 4*6*nmeasuredλ
	nC = (4+6*2)*nλ
	nelement = 4*6*(2*6+2)*nmeasuredλ+(4*6*2)*nmeasuredλ*(lk-1)
	L = zeros(Int,nelement)
	C = zeros(Int,nelement)
	V = zeros(T,nelement)
	c = 1

	for (i,baseline) ∈ enumerate(baselines)
		T1,T2 = baseline
		
		for  (ci,chnl) ∈ enumerate(["A","B","C","D"])
			wvlngth = get_wavelength(profiles["$T1$T2-$chnl-C"]; bnd=true)
			(mnw,mxw) =usable_wvlngth[i][:,ci]
			for (j,idx) ∈ enumerate(mnw:mxw)
				λ =  wvlngth[idx]
				isfinite(λ) || continue
				λidx = 	Float64.(find_index(λsampling,λ))
				λidx > 1 || continue
				# weights
				offweights = InterpolationKernels.compute_offset_and_weights(kernel, λidx)
				weights = vcat(offweights[2]...)
				off::Int = round(Int,offweights[1]) +1# + 1 a verifier
				mx = off + lk
				wsz::Int = lk
				if off <= 0 
					weights = weights[(2 - off):end]
					off = 1			
					weights = (sw=sum(weights))==0 ? weights : weights./sw
					wsz = length(weights)
				elseif (off+lk) > (nλ)
					mx = min(off + lk, nλ )
					weights = weights[1:(mx-off)] 
					weights = (sw=sum(weights))==0 ? weights : weights./sw
					wsz = length(weights)
				end

				rho = sqrt.(baseline_phasors[i][3,idx,ci].^2 .+ baseline_phasors[i][4,idx,ci].^2)
				coherence = rho .* exp.(-1im .* angle.(recalibrated_phasors[i][ci][idx]))
				Recoherence = real.(coherence)
				Imcoherence = imag.(coherence)
				#Recoherence = 2 .* real.(recalibrated_phasors[i][ci][idx])
				#Imcoherence = -2 .* imag.(recalibrated_phasors[i][ci][idx])

				# Interferometry
				#Real 
				V[c:(c+wsz-1)] .= weights.*Recoherence
				C[c:(c+wsz-1)] .= (((off:(off+wsz-1))).*(6*2+4)) .+ 4 .+ (i-1)*(2) .+ 1
				L[c:(c+wsz-1)] .= (( j-1)*(6*4)) .+ (i-1)*4 .+ ci
				c = c+wsz
				#Im
				V[c:(c+wsz-1)] .= weights.*Imcoherence 
				C[c:(c+wsz-1)] .= (((off:(off+wsz-1))).*(6*2+4)) .+ 4 .+ (i-1)*(2) .+ 2
				L[c:(c+wsz-1)] .= (( j-1)*(6*4)) .+ (i-1)*4 .+ ci
				c = c+wsz
				# photometry
			
				V[c:(c+wsz-1)] .= weights .*baseline_phasors[i][1,idx,ci]
				C[c:(c+wsz-1)] .= (((off:(off+wsz-1))).*(6*2+4)) .+ T1
				L[c:(c+wsz-1)] .= (( j-1)*(6*4)) .+ (i-1)*4 .+ ci
				c = c+wsz

				V[c:(c+wsz-1)] .= weights .*baseline_phasors[i][2,idx,ci]
				C[c:(c+wsz-1)] .= (((off:(off+wsz-1))).*(6*2+4)) .+ T2
				L[c:(c+wsz-1)] .= (( j-1)*(6*4)) .+ (i-1)*4 .+ ci
				c = c+wsz
			end 
		end
	end
	return sparse(L[1:c-1],C[1:c-1],V[1:c-1],nL,nC),λsampling,usable_wvlngth
end

build_interpolation_convolution_matrix((;knots,kernel)::Interpolator, samples,lsf) = build_interpolation_convolution_matrix(kernel, knots, samples, lsf) 


function build_interpolation_convolution_matrix(kernel::Kernel{T,N}, knots, samples,lsf) where {T,N}
	lin = length(samples)
	col = length(knots) 
	nkernel = length(kernel) 
	nK = nkernel * length(samples)
	#K = zeros(T,lin,col)
	KL = zeros(T,nK)
	KC = zeros(T,nK)
	KV = zeros(T,nK)

	nC = length(lsf)
	CL = zeros(T,nC)
	CC = zeros(T,nC)
	CV = zeros(T,nC)
	minl = minimum(axes(lsf,2))
	maxl = maximum(axes(lsf,2))

	iK = 1 
	iC = 1 
 	for (l,sample) ∈ enumerate(samples)
		offweights = InterpolationKernels.compute_offset_and_weights(kernel,T.(find_index(knots,sample))) 
		weights = vcat(offweights[2]...)
		off::Int = round(Int,offweights[1]) +1
		
		if ((off+N)>0 || off<=col ) 
			if off <= 0 
				weights = weights[(1 - off):end]
				off = 1			
				weights = (sw=sum(weights))==0 ? weights : weights./sw
			elseif (off+N) > col
				
				weights = weights[1:(col-off+1)] 
				weights = (sw=sum(weights))==0 ? weights : weights./sw
			end
		end
		wsz = length(weights)
		KL[iK:(iK+wsz-1)] .= l
		KC[iK:(iK+wsz-1)] .= off:(off+wsz-1)
		KV[iK:(iK+wsz-1)] .= weights
		iK = iK+wsz
		
		iCi = max(l + minl,1) 
		iCe = min(l + maxl,lin)
		CL[iC:(iC+iCe-iCi)] .= iCi:iCe
		CC[iC:(iC+iCe-iCi)] .= l
		CV[iC:(iC+iCe-iCi)] .= lsf[l,(iCi-l):(iCe-l)]
		iC = iC+iCe-iCi+1

		#K[l,off:(off+wsz-1)] .= weights
	end
	K = sparse(KL[1:iK-1],KC[1:iK-1],KV[1:iK-1],lin,col)
	C = sparse(CL[1:iC-1],CC[1:iC-1],CV[1:iC-1],lin,lin)
	return K, C
end

function build_interpolation_convolution_matrix(profile::SpectrumModel,(;knots,kernel)::Interpolator,hwdth::Int)
	λ = get_wavelength(profile; bnd=true)
	lsf = get_LSF(profile,hwdth)
	K, C = build_interpolation_convolution_matrix(kernel, knots, λ,lsf)
	return C * K
end