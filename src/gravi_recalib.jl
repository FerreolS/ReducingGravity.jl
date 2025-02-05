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
	return ML*gd
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

function recalibrate(data,opl,dispmodel, profiles::AbstractDict; baselines=baselines_list, iter=1)

	kt = [ [1. ./ get_wavelength(profiles["$(baseline[1])$(baseline[2])-$chnl-C"];bnd=true)  for (k,chnl) ∈ enumerate(["A","B","C","D"])] for (i,baseline) ∈ enumerate(baselines)]
	phasorst = [ Vector{Vector{ComplexF64}}(undef,4) for _ ∈ 1:6]
	for _ ∈ 1:iter
		kt,phasorst = recalibrate(data,opl,dispmodel, kt, phasorst; baselines=baselines)
	end
	return kt,phasorst
end

function recalibrate(data,opl,dispmodel, kt,phasorst; baselines=baselines_list)
	#Threads.@threads 
	for (i,baseline) ∈ collect(enumerate(baselines))
		T1,T2 = baseline
		#Threads.@threads
		for (j,chnl) ∈ collect(enumerate(["A","B","C","D"]))
			k = kt[i][j] 
			n1 = get_refractive_index(dispmodel[T1],k)
			n2 = get_refractive_index(dispmodel[T2],k)
			opd  = n1.*opl[T1]' .- n2 .* opl[T2]'
			d = view(data,j,i,:,:)
			phasorst[i][j] = recompute_phasors(d, k, opd)
			kt[i][j] = recalibrate_wavenumber(d, k, phasorst[i][j], opd)
		end
	end
	return kt,phasorst
end

function normalize_data(data, S, photometry; baselines=baselines_list)
	nλ = size(data,3)
	nt = size(data,4)
	nl = size(photometry[1],1)
	SS = reshape(S,4,6,nλ,16,nl)

	normalized = deepcopy(data)

	for (i,baseline) ∈ collect(enumerate(baselines))
		T1,T2 = baseline
		for (j,chnl) ∈ collect(enumerate(["A","B","C","D"]))
			d = view(data,j,i,:,:)
			photo1 = SS[j,i,:,T1,:]*photometry[T1]
			photo2 = SS[j,i,:,T2,:]*photometry[T2]
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