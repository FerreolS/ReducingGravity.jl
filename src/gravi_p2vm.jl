const baselines_list =[[1,2],[1,3],[4,1],[2,3],[4,2],[4,3]]
const triplet_list =[[1,2,3],[1,2,4],[1,3,4],[2,3,4]]
@enum Chnl chnlA=1 chnlB=2 chnlC=3 chnlD=4
Base.to_index(s::Chnl) = Int(s)

function  gravi_extract_channel(data::AbstractWeightedData{T,N},
								profile::SpectrumModel,
								lamp; kwds...)  where {T,N}
	chnl = gravi_extract_profile(data,profile; kwds...) 
	λ  = get_wavelength(profile;bnd=true)
	flux = lamp.(λ) 
	T1 = max.(T(0),profile.transmissions[1].(λ) .* flux)
	T2 = max.(T(0),profile.transmissions[2].(λ) .* flux)
	output =  (chnl - T1 -T2) / (2 .* sqrt.(T1 .* T2)) 
	for i ∈ findall(output.precision .<= 0)
		if i.I[1] == 1
			output.val[i] = output.val[i+ CartesianIndex(1,0)]
		elseif i.I[1] == size(output,1)
			output.val[i] = output.val[i- CartesianIndex(1,0)] 
		else
			output.val[i] = 0.5 .* (output.val[i- CartesianIndex(1,0)] + output.val[i+ CartesianIndex(1,0)])
		end
	end
	return output
end



function gravi_build_ABCD_phasors!(phasors,ϕ::AbstractArray{T,N},A,KA,B,KB,C,KC,D,KD;kwds...) where {T,N}
#= 
	for c in instances(Chnl)
		pA = build_phase(H,channels[c])
	end =#
	phasors[:,:,1] .= solve_phasor(ϕ,KA,A;kwds...)
	phasors[:,:,2] .= solve_phasor(ϕ,KB,B;kwds...)
	phasors[:,:,3] .= solve_phasor(ϕ,KC,C;kwds...)
	phasors[:,:,4] .= solve_phasor(ϕ,KD,D;kwds...)
	return phasors
end

function gravi_build_ABCD_phasors(ϕ::AbstractArray{T,N},A,KA,B,KB,C,KC,D,KD;kwds...) where {T,N}
	phasors = zeros(T,2,size(ϕ,1),4)
	gravi_build_ABCD_phasors!(phasors,ϕ,A,KA,B,KB,C,KC,D,KD;kwds...)
end


function solve_phasor(phi::AbstractArray{T,N}, 
					K,
					(;val, precision)::AbstractWeightedData;
					rgl_phasor = T(1e3),
					kwds...) where {T,N}
	if N==2
		H = similar(phi,size(K,1),2,size(phi,2))
		@tullio H[ll, 1, t] = cos(phi[c, t]) * K[ll, c]
		@tullio H[ll, 2, t] = sin(phi[c, t]) * K[ll, c]
	else
		@tullio H[ll, k, t] := phi[c, k, t] * K[ll, c]
	end

	@tullio C[ll, t,k,c] := H[ll, k,t] * K[ll, c]
	CC = reshape(C,size(C,1)*size(C,2),:)


	HtH = Symmetric(CC'*(precision[:].*CC) .+ rgl_phasor .* make_DtD(T,size(CC,2)))
	#HtH .+= diagm(1.e-3.*ones(size(HtH,1)))
	F = cholesky(HtH; check=false)
	if issuccess(F)
		v = F \ (CC'*(precision[:].*val[:]))
	else
		@show "phasors not cholesky $rgl_phasor"
		v = pinv(Array(HtH)) * (CC'*(precision[:].*val[:]))
	end
	@debug "phasor lkl : $(likelihood(WeightedData(val[:], precision[:]),CC*v )./length(val))"

	return  reshape(v,2, :)
end


function gravi_initial_input_phase(A,KA,B,KB,C,KC,D,KD)
	cA = compute_coefs(KA,A)
	cB = compute_coefs(KB,B)
	cC = compute_coefs(KC,C)
	cD = compute_coefs(KD,D)
    ϕ = atan.(cD.-cB,cC .-cA )

	return ϕ
end


function solve_visibility(phasors::AbstractArray{T,N},KA,KB,KC,KD,A,B,C,D;
							rgl_vis=T(1e3),
							kwds...) where {T,N}
	nl = size(KA,2)
	nt = size(A,2)
	visibilities = similar(phasors,nl,2,nt)
#	w_visibilities = similar(phasors,nl,2,nt)
	OA1 = KA.*phasors[1,:,1]'
	OA2 = KA.*phasors[2,:,1]'
	OA = hcat(OA1,OA2)
	b = OA'*(A.precision.*A.val)

	OB1 = KB.*phasors[1,:,2]'
	OB2 = KB.*phasors[2,:,2]'
	OB = hcat(OB1,OB2)
	b .+= OB'*(B.precision.*B.val)

	OC1 = KC.*phasors[1,:,3]'
	OC2 = KC.*phasors[2,:,3]'
	OC = hcat(OC1,OC2)
	b .+= OC'*(C.precision.*C.val)

	OD1 = KD.*phasors[1,:,4]'
	OD2 = KD.*phasors[2,:,4]'
	OD = hcat(OD1,OD2)
	b .+= OD'*(D.precision.*D.val)

	R = rgl_vis  .* make_DtD(T,size(OD,2)) 
	@debug "rgl_vis = $rgl_vis"
	@inbounds for t ∈ axes(A,2)
		HtH = OA'*(A.precision[:,t].*OA) 
		HtH .+= OB'*(B.precision[:,t].*OB) 
		HtH .+= OC'*(C.precision[:,t].*OC) 
		HtH .+= OD'*(D.precision[:,t].*OD) 
		HtH = Symmetric(HtH .+ R)
		#HtH .+= diagm(1.e-3.*ones(size(HtH,1)))
		F = cholesky(HtH; check=false)
    	if issuccess(F)
			#@show "Cholesky $t"
        	v = F \ b[:,t]
		#	w = diag(inv(F))
    	else
			@show "visibilities not cholesky"
			iHtH = pinv(HtH)
        	v = iHtH * b[:,t]
	#		w = diag(iHtH)
		end
		#v[.!isfinite.(v)].=T(0)
		visibilities[:,:,t] = reshape(v,nl,2)
	#	w_visibilities[:,:,t] = reshape(w,nl,2)
    end
	@debug "vis lkl : $(likelihood(A,OA* reshape(visibilities,size(OA,2),:))./length(A))"
	
	return visibilities #,w_visibilities

	
end


function gravi_build_p2vm_interf(p2vm_data::AbstractWeightedData{T,N},
								itrp::Interpolator, 
								profiles,
								lamp;
								loop=1 ,
								specres = 500,
								baselines=baselines_list,
								ptol=1e-5,
								rgl_phasor=T(1000),
								rgl_vis=T(1000),
								kwds...) where {T,N}

	baseline_phasors = Vector{Array{T,3}}(undef,6)
	baseline_visibilities = Vector{Array{T,3}}(undef,6)
	λ = itrp.knots
	Threads.@threads for (i,baseline) ∈ collect(enumerate(baselines))
		T1,T2 = baseline
		phi_sign = T1 > T2 ? -1 : 1
		A = gravi_extract_channel(p2vm_data,profiles["$T1$T2-A-C"],lamp)
		B = gravi_extract_channel(p2vm_data,profiles["$T1$T2-B-C"],lamp)
		C = gravi_extract_channel(p2vm_data,profiles["$T1$T2-C-C"],lamp)
		D = gravi_extract_channel(p2vm_data,profiles["$T1$T2-D-C"],lamp)

		KA = build_interpolation_matrix(itrp,get_wavelength(profiles["$T1$T2-A-C"];bnd=true))
		KB = build_interpolation_matrix(itrp,get_wavelength(profiles["$T1$T2-B-C"];bnd=true))
		KC = build_interpolation_matrix(itrp,get_wavelength(profiles["$T1$T2-C-C"];bnd=true))
		KD = build_interpolation_matrix(itrp,get_wavelength(profiles["$T1$T2-D-C"];bnd=true))

		ϕ = gravi_initial_input_phase(A,KA,B,KB,C,KC,D,KD)
		phasors= gravi_build_ABCD_phasors(ϕ,A,KA,B,KB,C,KC,D,KD;rgl_phasor=rgl_phasor)
		visibilities = solve_visibility(phasors,KA,KB,KC,KD,A,B,C,D;rgl_vis=rgl_vis)

		for _ ∈ 2:loop 
			prev = visibilities
			gravi_update_visibilities!(visibilities,λ;specres= specres, phi_sign=phi_sign)
			gravi_build_ABCD_phasors!(phasors,visibilities,A,KA,B,KB,C,KC,D,KD;rgl_phasor=rgl_phasor)
			visibilities = solve_visibility(phasors,KA,KB,KC,KD,A,B,C,D;rgl_vis=rgl_vis)
			sum(abs2,filter(isfinite,(visibilities.-prev))) < ptol && break
		end
		baseline_phasors[i] = phasors
		baseline_visibilities[i] = visibilities
	end
	return baseline_phasors, baseline_visibilities
end


function build_wavelength_range(profiles;
								padding=0, 
								λmin=0,
								λmax=1) 

	λstep = minimum([mean(diff(get_wavelength(p; bnd=true))) for p ∈ values(profiles)])

	wvmin = max(λmin,minimum([p.λbnd[1] for p ∈ values(profiles)]))
	wvmax = min(λmax,maximum([p.λbnd[2] for p ∈ values(profiles)]))

	return range(wvmin - padding * λstep,wvmax  +padding *  λstep; step = λstep)

end



function get_selected_wavelenght(profiles; 
								baselines=baselines_list,
								λmin=0,
								λmax=1) 
	
	usable_wvlngth = zeros(Int,2,6)
	for (i,baseline) ∈ enumerate(baselines)
		T1,T2 = baseline
		baseline_wvlngth = mean(hcat((get_wavelength(profiles["$T1$T2-$chnl-C"]; bnd=true) for chnl ∈["A","B","C","D"])...),dims=2)[:]
		fidx = findfirst(x->isfinite(x) && x>=λmin,baseline_wvlngth)
		lidx = findlast(x->isfinite(x) && x<=λmax,baseline_wvlngth)
		usable_wvlngth[:,i] .= [fidx,lidx]

	end
	return usable_wvlngth
end


function make_pixels_vector(data::AbstractWeightedData,
							profiles::AbstractDict,
							wvidx::Vector{Int}; 
							baselines=baselines_list, kwds...)  
	
	nframe = size(data,3)
	nmeasuredλ = wvidx[2] - wvidx[1]+1					
	nL = 4*6*nmeasuredλ
	v = zeros(Float64,nL,nframe)
	w = zeros(Float64,nL,nframe)
	for t ∈ axes(data,3)
		for (i,baseline) ∈ enumerate(baselines)
			T1,T2 = baseline
			for (k,chnl) ∈ enumerate(["A","B","C","D"])
				(;val,precision) = gravi_extract_profile(view(data,:,:,t),profiles["$T1$T2-$chnl-C"]; kwds...) 
				wvlngth = get_wavelength(profiles["$T1$T2-$chnl-C"])
				for (j,idx) ∈ enumerate(wvidx[1]:wvidx[2])
					isfinite(wvlngth[idx]) || isfinite(val[idx]) || continue
					v[((j-1)*(6*4)) + (i-1)*4 + k,t] = val[idx]
					w[((j-1)*(6*4)) + (i-1)*4 + k,t] = precision[idx]
				end
			end
		end
	end
	return WeightedData(v,w)
end

function extract_correlated_flux(x::AbstractArray{T,N};
								baselines=baselines_list) where {T,N}

	x = reshape(x,6*2+4,:,size(x,3))
	

	photometric= [x[i,:,:] for i∈1:4]
	interferometric = Vector{Matrix{Complex{T}}}(undef,6)
	
	for (i,baseline) ∈ enumerate(baselines)
		interferometric[i] = (x[4+2*(i-1)+1,..] .+ 1im .* x[4+2*(i-1)+2,..]) 
	end
	return photometric,interferometric
end




function gravi_build_V2PM(	profiles::AbstractDict,
									baseline_phasors; 
									baselines=baselines_list,
									λsampling=nothing,
									λmin=0.0,λmax=1.0,
									kernel = CatmullRomSpline())
	
	lk = length(kernel) 
	if isnothing(λsampling)
		λsampling =  build_wavelength_range(profiles;  padding=lk,λmin=λmin,λmax=λmax)
	end		
	λmin = max(minimum(λsampling),λmin)
	λmax = min(maximum(λsampling),λmax)
	usable_wvlngth = get_selected_wavelenght(profiles,baselines=baselines,λmin=λmin,λmax=λmax)
	nλ = length(λsampling)

	minwv = minimum(usable_wvlngth)
	maxwv = maximum(usable_wvlngth)
	nmeasuredλ = maxwv - minwv + 1

	nL = 4*6*nmeasuredλ
	nC = (4+6*2)*nλ
	nelement = 4*6*(2*6+2)*nmeasuredλ+(4*6*2)*nmeasuredλ*(lk-1)
	L = zeros(Int,nelement)
	C = zeros(Int,nelement)
	V = zeros(Float64,nelement)
	c = 1

	for (i,baseline) ∈ enumerate(baselines)
		T1,T2 = baseline
		for  (ci,chnl) ∈ enumerate(["A","B","C","D"])
			wvlngth = get_wavelength(profiles["$T1$T2-$chnl-C"])[:]
			for (j,idx) ∈ enumerate(minwv:maxwv)
				λ =  wvlngth[idx]
				isfinite(λ) || continue
				λidx = 	find_index(λsampling,λ)
				λidx > 1 || continue
				# weights
				off,weights = InterpolationKernels.compute_offset_and_weights(kernel, λidx)
				off = Int(off)+1
				mx = off + lk
				wsz = lk
				if off < 0 
					weights = weights[(1 - off):end]
					off = 0			
					weights = (sw=sum(weights))==0 ? weights : weights./sw
					wsz = length(weights)
				elseif (off+lk) > nC
					mx = min(off + lk, nC )
					weights = weights[1:(mx-off)] 
					weights = (sw=sum(weights))==0 ? weights : weights./sw
					wsz = length(weights)
				end

				trans1 = profiles["$T1$T2-$chnl-C"].transmissions[1](λ)
				trans2 = profiles["$T1$T2-$chnl-C"].transmissions[2](λ)

				# Interferometry
				trans =  weights.*sqrt(trans1*trans2) 
				#Real
				V[c:(c+wsz-1)] .= baseline_phasors[i][1,ci,idx] .* trans 
				C[c:(c+wsz-1)] .= (((off:(off+wsz-1))).*(6*2+4)) .+ 4 .+ (i-1)*(2) .+ 1
				L[c:(c+wsz-1)] .= (( j-1)*(6*4)) .+ (i-1)*4 .+ ci
				c = c+wsz
				#Im
				V[c:(c+wsz-1)] .= baseline_phasors[i][2,ci,idx] .* trans 
				C[c:(c+wsz-1)] .= (((off:(off+wsz-1))).*(6*2+4)) .+ 4 .+ (i-1)*(2) .+ 2
				L[c:(c+wsz-1)] .= (( j-1)*(6*4)) .+ (i-1)*4 .+ ci
				c = c+wsz
				# photometry
				#off,weights = InterpolationKernels.compute_offset_and_weights(kernel, λidx)
			
				V[c:(c+wsz-1)] .= weights .*trans1
				C[c:(c+wsz-1)] .= (((off:(off+wsz-1))).*(6*2+4)) .+ T1
				L[c:(c+wsz-1)] .= (( j-1)*(6*4)) .+ (i-1)*4 .+ ci
				c = c+wsz
				V[c:(c+wsz-1)] .= weights .*trans2
				C[c:(c+wsz-1)] .= (((off:(off+wsz-1))).*(6*2+4)) .+ T2
				L[c:(c+wsz-1)] .= (( j-1)*(6*4)) .+ (i-1)*4 .+ ci
				c = c+wsz
			end 
		end
	end
#	return V,C,L,c-1,nL,nC, nelement
#= 	L = L[1:c-1]
	C = C[1:c-1]
	V = V[1:c-1]
	minC = minimum(C)
	maxC = maximum(C)
	minL = minimum(L)
	maxL = maximum(L)
	@. L = L - minL+1
	@. C = C - minC+1
	nC = maxC - minC +1
	nL = maxL - minL +1 =#
	return sparse(L[1:c-1],C[1:c-1],V[1:c-1],nL,nC),λsampling[:],[minwv,maxwv]
end


function  get_correlatedflux(V2PM::AbstractMatrix{T},	
								data::AbstractWeightedData{T,2};
								maxeval=500,atol=1e-3) where {T}
		
	nframe = size(data,2)
	nrow = size(V2PM,2) ÷ (6*2+4)
	output = zeros(T,6*2+4,nrow,nframe)
	Threads.@threads for t ∈ 1:nframe
		#= A = Symmetric(V2PM'*spdiagm(view(precision,:,t))*V2PM)
		b = V2PM'*(view(precision,:,t) .* view(val,:,t))[:]	
		x,info= KrylovKit.linsolve(A,b; issymmetric=true,maxiter=maxiter,atol=atol,verbosity=1)
		  =#
		x = solveV2PM(V2PM, view(data,:,t),maxeval=maxeval)
		view(output,:,:,t)[:] .= x[:]
	end
	return output
end

function solveV2PM(V2PM, 
					(;val, precision)::AbstractWeightedData;
					maxeval=500) 

 	function fg!(x,g)
       r =(V2PM*x .- val)
       rp = precision.*r 
       g .= V2PM'*rp
       return sum(r.*rp)
	end
	x0 = V2PM'*val
	return vmlmb(fg!,  x0 ;maxeval=maxeval)
end

function  get_correlatedflux_rough(V2PM::AbstractMatrix{T},	
								(;val, precision)::AbstractWeightedData{T,2};
								kwds...) where {T}
		
	nframe = size(val,2)
	nrow = size(V2PM,2) ÷ (6*2+4)


	mwght = mean(precision,dims=2)
	Id = sparse(I,size(V2PM,2),size(V2PM,2))
	CxVt = pinv(Symmetric(Array(V2PM'*(mwght.*V2PM) .+ 1e-3.*Id)))*V2PM'
	
	output = zeros(T,6*2+4,nrow,nframe)
	Threads.@threads for t ∈ axes(val,2)
		view(output,:,:,t)[:] .= (CxVt*(mwght.* view(val,:,t)))[:]	
	end
	return output
end

function get_bispectrum(interferometric::AbstractVector{A};
					triplets=triplet_list,
					baselines = baselines_list) where{T,N,A<:AbstractArray{Complex{T},N}}

	bispectrum = Vector{Array{Complex{T},N}}(undef,length(triplets))
	for (i,triplet) ∈ enumerate(triplets)
		T1,T2,T3 = triplet
		b1 = findfirst(x->x== [T1,T2] ,baselines)
		if isnothing(b1)
			b1 = findfirst(x->x== [T2,T1] ,baselines)
			p1 = exp.(-1im.*angle.(interferometric[b1]))
		else
			p1 = exp.(1im.*angle.(interferometric[b1]))
		end
		b2 = findfirst(x->x== [T2,T3] ,baselines)
		if isnothing(b2)
			b2 = findfirst(x->x== [T3,T2] ,baselines_list)
			p2 = exp.(-1im.*angle.(interferometric[b2]))
		else
			p2 = exp.(1im.*angle.(interferometric[b2]))
		end

		b3 = findfirst(x->x== [T3,T1] ,baselines_list)
		if isnothing(b3)
			b3 = findfirst(x->x== [T1,T3] ,baselines_list)
			p3 = exp.(-1im.*angle.(interferometric[b3]))
		else
			p3 = exp.(1im.*angle.(interferometric[b3]))
		end
		bispectrum[i] = ( p1.*p2.*p3)
	end

	return bispectrum
end

function get_closures(  interferometric::AbstractVector{A};
						triplets=triplet_list,
						baselines = baselines_list) where{T,N,A<:AbstractArray{Complex{T},N}}
	return broadcast(x->angle.(x), get_bispectrum(interferometric; triplets=triplets, baselines = baselines))
end


function build_baselinecloseMatrix(;baselines = baselines_list, triplets=triplet_list)

	base2clos = zeros(4,6)
	for (i,triplet) ∈ enumerate(triplets)
		T1,T2,T3 = triplet
		b1 = findfirst(x->x== [T1,T2] ,baselines)
		if isnothing(b1)
			b1 = findfirst(x->x== [T2,T1] ,baselines)
			base2clos[i,b1] = -1
		else
			base2clos[i,b1] = 1
		end
		b2 = findfirst(x->x== [T2,T3] ,baselines)
		if isnothing(b2)
			b2 = findfirst(x->x== [T3,T2] ,baselines_list)
			base2clos[i,b2] = -1
		else
			base2clos[i,b2] = 1
		end
		b3 = findfirst(x->x== [T3,T1] ,baselines_list)
		if isnothing(b3)
			b3 = findfirst(x->x== [T1,T3] ,baselines_list)
			base2clos[i,b3] = -1
		else
			base2clos[i,b3] = 1
		end
	end
	return base2clos
end

function zeroclosure!(interferometric, closures;baselines = baselines_list, triplets=triplet_list)
	M = build_baselinecloseMatrix(;baselines = baselines, triplets=triplets)
	C = dropdims(mean(cat(closures...,dims=3),dims=2),dims=2)
	for l ∈ axes(C,1)
		basephi = (1/4 .*M'*C[l,:])
		for b ∈ axes(interferometric,1)
			iphi = exp(-1im* basephi[b])
			interferometric[b][l,:]  .*= iphi
		end
	end
end


function mean_opd_create(visibilities::AbstractArray{T,3}, λ; phi_sign=1) where T
	ϕ = atan.(visibilities[:,2,:],visibilities[:,1,:])
	r = sqrt.(abs2.(visibilities[:,1,:]).+abs2.(visibilities[:,2,:]))
	lmin = maximum(mapslices(s->findfirst(x->  0.9 .< x .< 1.1,s),r,dims=1))
	lmax = minimum(mapslices(s->findlast(x->  0.9 .< x .< 1.1,s),r,dims=1))
	unwrap!(ϕ)
	opd = phi_sign .* mean((T.(λ ./(2π)) .* ϕ)[lmin:lmax,:],dims=1)[:]
	gd = angle.(mean(exp.(1im .* diff(ϕ[lmin:lmax,:],dims=1)),dims=1))[:]
	intercept, _ =  affine_solve(opd,gd)
	opd .-= intercept
	return opd
end


function mean_opd_create(ϕ::AbstractArray{T,2}, λ; phi_sign=1) where T
	unwrap!(ϕ)
	opd = phi_sign .*  mean((T.(λ ./(2π)) .* ϕ)[lmin:lmax,:],dims=1)[:]
	gd = angle.(mean(exp.(1im .* diff(ϕ[lmin:lmax,:],dims=1)),dims=1))[:]
	intercept, _ =  affine_solve(opd,gd)
	opd .-= intercept
	return opd
end


function gravi_compute_envelope(opd::AbstractVector{T}, λ) where {T}
	nλ = length(λ)
	nt = length(opd)

	# Compute delta_lambda and lambda from experience 
	Δλ = mean(diff(λ))*3


	coh_len = @.  T(λ^2 / Δλ)

	# Gaussian enveloppe 
	envellope = zeros(T,nλ,nt)
	for idx ∈ CartesianIndices(envellope)
		i, j = Tuple(idx)
		envellope[i,j] = exp(-1*(opd[j]^2)/(coh_len[i]^2/2))
	end
	return envellope
end



function gravi_compute_envelope(opd::AbstractVector{T}, λ,R) where {T}
	nλ = length(λ)
	nt = length(opd)

	coh_len = @. (2log(2)) / π * T(λ * R)

	# Gaussian enveloppe 
	envellope = zeros(T,nλ,nt)
	for idx ∈ CartesianIndices(envellope)
		i, j = Tuple(idx)
		envellope[i,j] = exp(-1*(opd[j]^2)/(coh_len[i]^2/2))
	end
	return envellope
end

function unwrap!(ϕ::AbstractMatrix{T}; period = 2π) where T 
	unwrap!(view(ϕ,:,1))
	for x = eachslice(ϕ,dims=1)
		unwrap!(view(x,:))
	end
end

function gravi_update_visibilities!(visibilities,λ; phi_sign = 1, specres=500, )
	opd = mean_opd_create(visibilities, λ; phi_sign= phi_sign)
	envellope = gravi_compute_envelope(opd,λ,specres)
	ϕ = opd' .* (2π ./ λ)
	visibilities[:,1,:] .= envellope .* cos.(ϕ)
	visibilities[:,2,:] .= envellope .* sin.(ϕ)

end