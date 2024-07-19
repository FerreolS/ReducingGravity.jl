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

function gravi_estimate_ABCD_phasor(H::AbstractArray{T,N},channels) where {T,N}
	if N==1
		cϕ = cos.(H)
		sϕ = sin.(H) # sign?
		H = hcat(cϕ,sϕ)
	end
	#H[:,2] .*=-1
	for c in instances(Chnl)
		pA = build_phase(H,channels[c])
	end

	pA = build_phase(H,A)
	pB = build_phase(H,B)
	pC = build_phase(H,C)
	pD = build_phase(H,D)
	return (pA,pB,pC,pD)
end

function gravi_build_ABCD_phasors!(phasors,ϕ::AbstractArray{T,N},A,KA,B,KB,C,KC,D,KD;kwds...) where {T,N}

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
					rgl_phasor = T(1e-3),
					kwds...) where {T,N}
	if N==2
		H = similar(phi,size(K,1),2,size(phi,2))
		#= @inbounds @simd for j ∈ axes(phi,2)
			sp = similar(phi,size(K,2))
			cp = similar(phi,size(K,2))
			@inbounds @simd for i ∈ axes(phi,1)
				sp[i], cp[i] = sincos(phi[i,j])
			end
			H[:,1,:] .= K * cp
			H[:,2,:] .= K * sp
		end =#
		@tullio H[ll, 1, t] = cos(phi[c, t]) * K[ll, c]
		@tullio H[ll, 2, t] = sin(phi[c, t]) * K[ll, c]
	else
		@tullio H[ll, k, t] := phi[c, k, t] * K[ll, c]
	end

	#@tullio C[k, c, ll, t] := H[ll, t, k] * K[ll, c]
	@tullio C[ll, t,k,c] := H[ll, k,t] * K[ll, c]
	CC = reshape(C,size(C,1)*size(C,2),:)


	HtH = Symmetric(CC'*(precision[:].*CC) .+ rgl_phasor .* make_DtD(T,size(CC,2)))
	#HtH .+= diagm(1.e-3.*ones(size(HtH,1)))
	F = cholesky(HtH; check=false)
	if issuccess(F)
		v = F \ (CC'*(precision[:].*val[:]))
	else

		@show "phasors not cholesky $rgl"
		v = pinv(Array(HtH)) * (CC'*(precision[:].*val[:]))
	end
	@debug "phasor lkl : $(likelihood(WeightedData(val[:], precision[:]),CC*v )./length(val))"

	#v[.!isfinite.(v)] .= T(0)

	#return reshape(pinv(CC'*(precision[:].*CC))*CC'*(precision[:].*val[:]),2, :)
	return  reshape(v,2, :)
end

function solve_phasor_opt(phi::AbstractArray{T,N}, 
					K,
					(;val, precision)::AbstractWeightedData,
					xinit;
					kwds...) where {T,N}
	if N==2
		H = cat(K*cos.(phi),K*sin.(phi),dims=3)
	else
		@tullio H[ll, t, k] := phi[c, t, k] * K[ll, c]
	end

	@tullio C[k, c, ll, t] := H[ll, t, k] * K[ll, c]
	d = similar(val)
 	function fg!(x,g)
		@tullio d[ll,t] = C[k,c,ll,t] * x[k,c]
       r =(d .- val)
       rp = precision.*r 
	   @tullio g[k,c] = C[k,c,ll,t] * rp[ll,t]
       return sum(r.*rp)
	end
	# @tullio x0[k,c] := C[ k,c,ll,t] * val[ll,t]
	return vmlmb(fg!,  xinit ;kwds...)
end
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

function gravi_initial_input_phase(A,KA,B,KB,C,KC,D,KD)
	cA = compute_coefs(KA,A)
	cB = compute_coefs(KB,B)
	cC = compute_coefs(KC,C)
	cD = compute_coefs(KD,D)
    ϕ = atan.(cD.-cB,cC .-cA )

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



function solve_visibility_opt(visibilities,phasors,KA,KB,KC,KD,A,B,C,D;maxeval=100,kwds...)
	OA1 = KA.*phasors[1,:,1]'
	OA2 = KA.*phasors[2,:,1]'

	OB1 = KB.*phasors[1,:,2]'
	OB2 = KB.*phasors[2,:,2]'

	OC1 = KC.*phasors[1,:,3]'
	OC2 = KC.*phasors[2,:,3]'

	OD1 = KD.*phasors[1,:,4]'
	OD2 = KD.*phasors[2,:,4]'
	#phase =zeros(Float64,2,size(KA,2),size(A,2))


 	function fg!(x,g)
		r = OA1*x[:,1,:] .+ OA2*x[:,2,:] .- A.val 
		rp = A.precision.*r
		g[:,1,:] .= OA1'*rp
		g[:,2,:] .= OA2'*rp
		cost = sum(r.*rp)

		r = OB1*x[:,1,:] .+ OB2*x[:,2,:] .- B.val 
		rp = B.precision.*r
		g[:,1,:] .+= OB1'*rp
		g[:,2,:] .+= OB2'*rp
		cost += sum(r.*rp)

		r =  OC1*x[:,1,:] .+ OC2*x[:,2,:] .- C.val 
		rp = C.precision.*r
		g[:,1,:] .+= OC1'*rp
		g[:,2,:] .+= OC2'*rp
		cost += sum(r.*rp)

		r =  OD1*x[:,1,:] .+ OD2*x[:,2,:] .- D.val 
		rp = D.precision.*r
		g[:,1,:] .+= OD1'*rp
		g[:,2,:] .+= OD2'*rp
		cost += sum(r.*rp)

       return cost
	end

	function f(x)
		r = A.val .- OA1*x[:,:,1] .- OA2*x[:,:,2]
		rp = A.precision.*r
		costA = sum(r.*rp)

		r = B.val .- OB1*x[:,:,1] .- OB2*x[:,:,2]
		rp = B.precision.*r
		costB = sum(r.*rp)

		r = C.val .- OC1*x[:,:,1] .- OC2*x[:,:,2]
		rp = C.precision.*r
		costC = sum(r.*rp)

		r = D.val .- OD1*x[:,:,1] .- OD2*x[:,:,2]
		rp = D.precision.*r
		costD = sum(r.*rp)

       return costA+costB+costC+costD
	end
	return vmlmb(fg!,  visibilities ;maxeval=maxeval,kwds...	)
	#return vmlmb(f,  xinit ;autodiff=true,maxeval=maxeval,verb=verb,xtol=(0.,0.)	)

	
end



function solve_visibility(phasors::AbstractArray{T,N},KA,KB,KC,KD,A,B,C,D;
							rgl_vis=1e-3,
							kwds...) where {T,N}
	nl = size(KA,2)
	nt = size(A,2)
	visibilities = similar(phasors,nl,2,nt)
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

	R = Array(rgl_vis  .* make_DtD(T,size(OD,2)) )
	@debug "rgl_vis = $rgl_vis"
	@inbounds for t ∈ axes(A,2)
		HtH = OA'*(A.precision[:,t].*OA) 
		HtH .+= OB'*(B.precision[:,t].*OB) 
		HtH .+= OC'*(C.precision[:,t].*OC) 
		HtH .+= OD'*(D.precision[:,t].*OD) 
		HtH = (Symmetric(HtH) .+  R)
		#HtH .+= diagm(1.e-3.*ones(size(HtH,1)))
		F = cholesky(HtH; check=false)
    	if issuccess(F)
			#@show "Cholesky $t"
        	v = F \ b[:,t]
    	else
			@show "visibilities not cholesky"
        	v = pinv(HtH) * b[:,t]
		end
		#v[.!isfinite.(v)].=T(0)
		visibilities[:,:,t] = reshape(v,nl,2)
    end
	@debug "vis lkl : $(likelihood(A,OA* reshape(visibilities,size(OA,2),:))./length(A))"
	
	return visibilities

	
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

function gravi_build_p2vm_interf(p2vm_data::AbstractWeightedData{T,N},
								itrp, 
								profiles,
								lamp;
								loop_with_norm=1,loop=1 ,baselines=baselines_list,ptol=1e-5,kwds...) where {T,N}

	baseline_phasors = Vector{Array{T,3}}(undef,6)
	baseline_visibilities = Vector{Array{T,3}}(undef,6)

	Threads.@threads for (i,baseline) ∈ collect(enumerate(baselines))
		T1,T2 = baseline
		A = gravi_extract_channel(p2vm_data,profiles["$T1$T2-A-C"],lamp)
		B = gravi_extract_channel(p2vm_data,profiles["$T1$T2-B-C"],lamp)
		C = gravi_extract_channel(p2vm_data,profiles["$T1$T2-C-C"],lamp)
		D = gravi_extract_channel(p2vm_data,profiles["$T1$T2-D-C"],lamp)

		KA = build_interpolation_matrix(itrp,get_wavelength(profiles["$T1$T2-A-C"]))
		KB = build_interpolation_matrix(itrp,get_wavelength(profiles["$T1$T2-B-C"]))
		KC = build_interpolation_matrix(itrp,get_wavelength(profiles["$T1$T2-C-C"]))
		KD = build_interpolation_matrix(itrp,get_wavelength(profiles["$T1$T2-D-C"]))

		ϕ = gravi_initial_input_phase(A,KA,B,KB,C,KC,D,KD)
		nt = size(ϕ,2)
		phasors= gravi_build_ABCD_phasors(ϕ,A,KA,B,KB,C,KC,D,KD;kwds...)
		#visibilities = cat(cos.(ϕ) ,sin.(ϕ),dims=3)

		#visibilities = solve_visibility_opt(visibilities,phasors,KA,KB,KC,KD,A,B,C,D;kwds...)
		visibilities = solve_visibility(phasors,KA,KB,KC,KD,A,B,C,D;kwds...)
		#coherence = ones(T,nt)
		#rho3 = similar(ϕ,size(ϕ)[1:2])
		for _ ∈ 1:loop_with_norm
			#rho3 = (ones(360) .* median(rho,dims=1))
			#rho3 =  ones(size(visibilities,1)) .* median(rho,dims=1)
			gravi_build_ABCD_phasors!(phasors,visibilities,A,KA,B,KB,C,KC,D,KD;kwds...)
			visibilities = solve_visibility(phasors,KA,KB,KC,KD,A,B,C,D;kwds...)
			rho = sqrt.(sum(abs2,visibilities,dims=2))
			#= for i ∈ axes(ϕ,2)
				coherence = fit_envellope(rho[30:230,i],itrp.knots[30:230].*1e6)
				rho3[:,i] = coherence[2] .* envellope(coherence[1],itrp.knots.*1e6)
			end  =#
			rho3 =  ones(size(visibilities,1)) .* median(rho,dims=1)
			visibilities .*= 1 ./ rho  .* rho3
			visibilities[.!isfinite.(visibilities)] .= T(0)

		end
		for _ ∈ 1:loop  
			prev = visibilities
			gravi_build_ABCD_phasors!(phasors,visibilities,A,KA,B,KB,C,KC,D,KD;kwds...)
			visibilities = solve_visibility(phasors,KA,KB,KC,KD,A,B,C,D;kwds...)
			sum(abs2,filter(isfinite,(visibilities.-prev))) < ptol && break
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
envellope(value,rngλ) = exp.(-1/2 .* (value ./ rngλ).^2)

function fit_envellope(modulus::AbstractArray{T,N},rngλ; xinit=[T(1),T(1)],kwds...) where {T,N}
	function f(x)
		return sum(abs2,x[2].*envellope(x[1],rngλ)  .- modulus)
	end
   # @tullio x0[k,c] := C[ k,c,ll,t] * val[ll,t]
	return vmlmb(f,  xinit ;autodiff=true,kwds...)
end

function build_wavelength_range(profiles;
								padding=0, 
								λmin=0,
								λmax=1) 

	λstep = minimum([mean(diff(get_wavelength(p))) for p ∈ values(profiles)])

	wvmin = max(λmin,minimum([p.λbnd[1] for p ∈ values(profiles)]))
	wvmax = min(λmax,maximum([p.λbnd[2] for p ∈ values(profiles)]))

	return range(wvmin - padding * λstep,wvmax  +padding *  λstep; step = λstep)

end

#= 

function wavelength_range(profiles; 
							baselines=baselines_list,
							padding=0, 
							λmin=0,
							λmax=1) 
	λstep = minimum([mean(diff(ReducingGravity.get_wavelength(p))) for p ∈ values(profiles)])
	wvmin = 1
	wvmax = 0

	for (i,baseline) ∈ enumerate(baselines)
		T1,T2 = baseline

		wvmin = max(λmin,min(wvmin,minimum([profiles["$T1$T2-$chnl-C"].λbnd[1] for chnl ∈["A","B","C","D"]])))
		wvmax = min(λmax,max(wvmax,maximum([profiles["$T1$T2-$chnl-C"].λbnd[2] for chnl ∈["A","B","C","D"]])))
	end

	usable_wvlngth = get_selected_wavelenght(profiles,baselines=baselines,λmin=wvmin,λmax=wvmax)


	return range(wvmin - padding * λstep,wvmax  +padding *  λstep; step = λstep), usable_wvlngth
end =#

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