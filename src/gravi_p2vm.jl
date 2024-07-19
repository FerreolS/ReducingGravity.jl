const baselines_list =[[1,2],[1,3],[4,1],[2,3],[4,2],[4,3]]
const triplet_list =[[1,2,3],[1,2,4],[1,3,4],[2,3,4]]

function  gravi_extract_channel(data::AbstractWeightedData{T,N},
								profile::SpectrumModel,
								lamp; kwds...)  where {T,N}
	chnl = gravi_extract_profile(data,profile; kwds...) 
	λ  = get_wavelength(profile)
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
#= 
function gravi_estimate_ABCD_phasor(ϕ::AbstractArray{T,1},A,B,C,D) where T
	cϕ = cos.(ϕ)
	sϕ = sin.(ϕ) # sign?
	H = hcat(cϕ,sϕ)
	pA = build_phase(H,A)
	pB = build_phase(H,B)
	pC = build_phase(H,C)
	pD = build_phase(H,D)
	return (pA,pB,pC,pD)
end =#

function gravi_estimate_ABCD_phasor(H::AbstractArray{T,N},A,B,C,D) where {T,N}
	if N==1
		cϕ = cos.(H)
		sϕ = sin.(H) # sign?
		H = hcat(cϕ,sϕ)
	end
	#H[:,2] .*=-1
	pA = build_phase(H,A)
	pB = build_phase(H,B)
	pC = build_phase(H,C)
	pD = build_phase(H,D)
	return (pA,pB,pC,pD)
end

function gravi_initial_input_phase(A,B,C,D)
    ϕ = atan.((D-B).val,(C -A).val )
	err = (A.precision .<=0 .||  B.precision .<=0 .||  C.precision .<=0 .||  D.precision .<=0 )
	for i ∈ findall(err)
		ϕ[i] =  ϕ[i- CartesianIndex(1,0)]
	end
	return ϕ
end

function build_phase(H,data)
	p  = data.precision
	if all(iszero,p)
		return zeros(Float64,2)
	end
	return inv(H'*(p.*H))*H'*(p .*data.val)
end

#= function  gravi_build_ABCD_phasors(ϕ::AbstractArray{T,2},A,B,C,D; zeroA=false) where T
	phasors = zeros(Float64,2,4,size(ϕ,1))
	for l ∈ axes(ϕ,1)
		phi,a,b,c,d = view(ϕ,l,:),view(A,l,:),view(B,l,:),view(C,l,:),view(D,l,:)
		if any(isnan,phi) || any(iszero,phi)
			continue
		end
		(pA,pB,pC,pD) = gravi_estimate_ABCD_phasor(phi,a,b,c,d)
		#= if zeroA
			cpA = complex(-pA...) 
			cpA = cpA * exp(-1im * angle(cpA))
			pA = [real(cpA), imag(cpA)]
		end =#
		phasors[:,1,l] .= pA[:]
		phasors[:,2,l] .= pB[:]
		phasors[:,3,l] .= pC[:]
		phasors[:,4,l] .= pD[:]
	end
	return phasors
end
 =#
function gravi_build_ABCD_phasors(ϕ::AbstractArray{T,N},A,B,C,D) where {T,N}
	if N==2
		nl = size(ϕ,1)
	else
		nl = size(ϕ,2)
	end
	phasors = zeros(Float64,2,4,nl)

	for l ∈ 1:nl
		if N==2
			phi = view(ϕ,l,:)
		else
			phi = view(ϕ,:,l,:)'
		end
		a,b,c,d = view(A,l,:),view(B,l,:),view(C,l,:),view(D,l,:)
		if any(.!isfinite,phi) || any(iszero,phi)
			continue
		end
		(pA,pB,pC,pD) = gravi_estimate_ABCD_phasor(phi,a,b,c,d)
		phasors[:,1,l] .= pA[:]
		phasors[:,2,l] .= pB[:]
		phasors[:,3,l] .= pC[:]
		phasors[:,4,l] .= pD[:]
	end
	return phasors
end

function estimate_visibility(phasors,A,B,C,D;
						robust=false)
	phase =zeros(Float64,2,size(A)...)
	for l ∈ axes(A,1)
		P = phasors[:,:,l]'
		#P[:,2] .*=-1
		#if any(x->(iszero(x)||isnan(x)),P[:]) 
		if any(x->(isnan(x)),P) || sum(iszero.(P))>2
			continue
		end

		for t ∈ axes(A,2)
			input = [A.val[l,t] ;B.val[l,t] ;C.val[l,t]; D.val[l,t]]
			w = [A.precision[l,t];B.precision[l,t];C.precision[l,t];D.precision[l,t]]
			if sum(iszero.(w))>2 || any(isnan,w[:])
				continue #phase[l,t,:] .=  zeros(Float64,2)
			else
				phase[:,l,t] .= (inv(P'*(w.*P))*P'*(w.*input))[:]
				if robust
					res = sqrt.(w) .* (P * phase[l,t,:] .- input) 
					w .*= (-2.795 .< res .<  2.795)
					if  sum(iszero.(w))>2 || any(isnan,w[:])
						continue
					end
					phase[:,l,t] .= (inv(P'*(w.*P))*P'*(w.*input))[:]
				end
			end
		end
	end
	return phase
end

function gravi_build_p2vm_interf(p2vm_data,profiles,lamp;loop_with_norm=5,loop=5,baselines=baselines_list,ptol=1e-5)

	baseline_phasors = Vector{Array{Float64,3}}(undef,6)
	baseline_visibilities = Vector{Array{Float64,3}}(undef,6)

	for (i,baseline) ∈ enumerate(baselines)
		T1,T2 = baseline
		A = gravi_extract_channel(p2vm_data,profiles["$T1$T2-A-C"],lamp)
		B = gravi_extract_channel(p2vm_data,profiles["$T1$T2-B-C"],lamp)
		C = gravi_extract_channel(p2vm_data,profiles["$T1$T2-C-C"],lamp)
		D = gravi_extract_channel(p2vm_data,profiles["$T1$T2-D-C"],lamp)
		ϕ = gravi_initial_input_phase(A,B,C,D)
		phasors= gravi_build_ABCD_phasors(ϕ,A,B,C,D)
		visibilities = estimate_visibility(phasors,A,B,C,D)
		for _ ∈ 1:loop_with_norm
			rho = sqrt.(visibilities[1,:,:].^2 .+ visibilities[2,:,:] .^2)
			#rho3 = (ones(360) .* median(rho,dims=1))
			rho3 =  ones(360) .* median(rho[50:200,:],dims=1)
			visibilities .*= reshape(1 ./ rho  .* rho3,1,size(rho)...)
			phasors= gravi_build_ABCD_phasors(visibilities,A,B,C,D);
			visibilities = estimate_visibility(phasors,A,B,C,D);
		end
		for _ ∈ 1:loop
			prev = visibilities
			phasors= gravi_build_ABCD_phasors(visibilities,A,B,C,D);
			visibilities = estimate_visibility(phasors,A,B,C,D);
			sum(abs2,filter(isfinite,(visibilities.-prev))) < ptol || break
		end
		baseline_phasors[i] = phasors
		baseline_visibilities[i] = visibilities
	end
	return baseline_phasors, baseline_visibilities
end

#= /* Compute delta_lambda and lambda from experience */
	double delta_lambda = (nwave > GRAVI_LBD_FTSC) ? 0.45 / nwave * 3 : 0.13;
    double lambda = 2.0 + 0.45 / nwave * wave;
    
	 /* Compute coherent length */
	double coh_len= (lambda*lambda) / delta_lambda * 1.e-6;

	long nrow = cpl_vector_get_size (opd);
	cpl_vector * envelope = cpl_vector_new (nrow);

	/* Gaussian enveloppe */
	for (long row = 0; row < nrow; row++){
		double value = cpl_vector_get (opd, row);
        cpl_vector_set (envelope, row, exp(-1*(value*value)/(coh_len*coh_len/2.)));
        CPLCHECK_NUL ("Cannot compute envelope");
	} =#
function fit_envellope(visibilities)
	
	
end
function wavelength_range(profiles; 
							baselines=baselines_list,
							padding=0, 
							λmin=0,
							λmax=1) 
	λstep = minimum([mean(diff(ReducingGravity.get_wavelength(p; bnd=false))) for p ∈ values(profiles)])
	wvmin = 1
	wvmax = 0

	for (i,baseline) ∈ enumerate(baselines)
		T1,T2 = baseline

		wvmin = max(λmin,min(wvmin,minimum([profiles["$T1$T2-$chnl-C"].λbnd[1] for chnl ∈["A","B","C","D"]])))
		wvmax = min(λmax,max(wvmax,maximum([profiles["$T1$T2-$chnl-C"].λbnd[2] for chnl ∈["A","B","C","D"]])))
	end

	usable_wvlngth = get_selected_wavelenght(profiles,baselines=baselines,λmin=wvmin,λmax=wvmax)


	return range(wvmin - padding * λstep,wvmax  +padding *  λstep; step = λstep), usable_wvlngth
end

function get_selected_wavelenght(profiles; 
								baselines=baselines_list,
								λmin=0,
								λmax=1) 
	
	usable_wvlngth = zeros(Int,2,6)
	for (i,baseline) ∈ enumerate(baselines)
		T1,T2 = baseline
		baseline_wvlngth = mean(hcat((get_wavelength(profiles["$T1$T2-$chnl-C"]) for chnl ∈["A","B","C","D"])...),dims=2)[:]
		fidx = findfirst(x->isfinite(x) && x>=λmin,baseline_wvlngth)
		lidx = findlast(x->isfinite(x) && x<=λmax,baseline_wvlngth)
		usable_wvlngth[:,i] .= [fidx,lidx]

	end
	return usable_wvlngth
end

function gravi_build_p2vm_matrix(	profiles,
									baseline_phasors; 
									baselines=baselines_list,
									λmin=0.0,λmax=1.0,
									kernel = first(values(profiles)).transmissions[1].basis.kernel)

	λbaseline = Vector{Vector{Float64}}(undef,6)
	
	lk = length(kernel) 
	tλ,usable_wvlngth = wavelength_range(profiles; baselines=baselines, padding=lk,λmin=λmin,λmax=λmax)
	nλ = length(tλ)

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
		baseline_wvlngth = mean(hcat((get_wavelength(profiles["$T1$T2-$chnl-C"]) for chnl ∈["A","B","C","D"])...),dims=2)[:]
		λbaseline[i] = baseline_wvlngth[minwv-lk:maxwv+lk]
		for (j,idx) ∈ enumerate(minwv:maxwv)
			λ =  baseline_wvlngth[idx]
			isfinite(λ) || continue
			λidx = 	find_index(tλ,λ)
			λidx > 1 || continue
			# Interferometry
			trans =  [sqrt(profiles["$T1$T2-$chnl-C"].transmissions[1](λ).*profiles["$T1$T2-$chnl-C"].transmissions[2](λ)) for chnl in ["A","B","C","D"]]
			V[c:c+7] .= baseline_phasors[i][idx,:,:][:] .* trans[ [1,2,3,4,1,2,3,4] ] #.* [1 ,1 ,1 ,1, -1, -1, -1, -1]
			C[c:c+7] .= (( round(Int,λidx)-1)*(6*2+4)) .+ 4 .+ (i-1)*(2) .+ [1,1,1,1,2,2,2,2]
			L[c:c+7] .= (( j-1)*(6*4)) .+ (i-1)*4 .+ [1,2,3,4,1,2,3,4] # [1,1,2,2,3,3,4,4]
			c = c+8
			# photometry
			off,weights = InterpolationKernels.compute_offset_and_weights(kernel, λidx)
			
			off = Int(off)
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
			#kλ = view(tλ,off:(off+lk-1))
			for (k,chnl) ∈enumerate(["A","B","C","D"])
				trans1 = profiles["$T1$T2-$chnl-C"].transmissions[1](λ)
				trans2 = profiles["$T1$T2-$chnl-C"].transmissions[2](λ)
				#@show size(V[c:(c+lk-1)])
				#@show size(weights.*trans1)
				V[c:(c+wsz-1)] .= weights .*trans1
				C[c:(c+wsz-1)] .= (((off:(off+wsz-1))).*(6*2+4)) .+ T1
				L[c:(c+wsz-1)] .= (( j-1)*(6*4)) .+ (i-1)*4 .+ k
				c = c+wsz
				V[c:(c+wsz-1)] .= weights .*trans2
				C[c:(c+wsz-1)] .= (((off:(off+wsz-1))).*(6*2+4)) .+ T2
				L[c:(c+wsz-1)] .= (( j-1)*(6*4)) .+ (i-1)*4 .+ k
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
	return sparse(L[1:c-1],C[1:c-1],V[1:c-1],nL,nC),tλ[:],λbaseline,[minwv,maxwv]
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
		λsampling,usable_wvlngth = wavelength_range(profiles; baselines=baselines, padding=lk,λmin=λmin,λmax=λmax)
		λmin = min(minimum(λsampling),λmin)
		λmax = min(minimum(λsampling),λmax)
	else
		λmin = min(minimum(λsampling),λmin)
		λmax = min(minimum(λsampling),λmax)
		usable_wvlngth = get_selected_wavelenght(profiles,baselines=baselines,λmin=λmin,λmax=λmax)
	end
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
								maxiter=100,atol=1e-3) where {T}
		
	nframe = size(data,2)
	nrow = size(V2PM,2) ÷ (6*2+4)
	output = zeros(T,6*2+4,nrow,nframe)
	Threads.@threads for t ∈ 1:nframe
		#= A = Symmetric(V2PM'*spdiagm(view(precision,:,t))*V2PM)
		b = V2PM'*(view(precision,:,t) .* view(val,:,t))[:]	
		x,info= KrylovKit.linsolve(A,b; issymmetric=true,maxiter=maxiter,atol=atol,verbosity=1)
		  =#
		x = solveV2PM(V2PM, view(data,:,t))
		view(output,:,:,t)[:] .= x[:]
	end
	return output
end

function solveV2PM(V2PM, 
					(;val, precision)::AbstractWeightedData) 

 	function fg!(x,g)
       r =(V2PM*x .- val)
       rp = precision.*r 
       g .= V2PM'*rp
       return sum(r.*rp)
	end
	x0 = V2PM'*val
	return vmlmb(fg!,  x0 ;maxeval=500)
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