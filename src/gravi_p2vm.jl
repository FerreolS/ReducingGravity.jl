const baselines_list =[[1,2],[1,3],[4,1],[2,3],[4,2],[4,3]]

function gravi_create_single_baseline(	tel1::Int,
										tel2::Int,
										data::AbstractWeightedData{T,N},
										dark::AbstractWeightedData,
										lamp::AbstractVector,
										profiles::SpectrumModel) where {T,N}

	extracted_spectra = Dict{String,Vector{WeightedData{Float64,1, Vector{Float64}, Vector{Float64}}}}()
	for (key,profile) ∈ profiles
		push!(extracted_spectra,key=>[gravi_extract_profile(view(data,:,:,frame) - dark,profile) for frame ∈ axes(data,3)])
	end
	return extracted_spectra

	ABCD = ConcreteWeightedData{T,N}
	for (key,profile) ∈ profiles
		if (tel1 == key[1]) && (tel2 == key[2])
			continue
		end
		key1 = "$tel1-$key" 
		key2 = "$tel2-$key" 
	end
	for chnl ∈ ["A","B","C","D"]
		profiles["$tel1$tel2-$chnl-C"]
	end	
end


function  gravi_extract_channel(data::AbstractWeightedData,
								profile::SpectrumModel,
								lamp; kwds...) 
	chnl = gravi_extract_profile(data,profile; kwds...) 
	λ  = get_wavelength(profile)
	flux = lamp.(λ) 
	T1 = max.(0.,profile.transmissions[1].(λ) .* flux)
	T2 = max.(0.,profile.transmissions[2].(λ) .* flux)
	output =  (chnl - T1 -T2) / (2 .* sqrt.(T1 .* T2)) 
	for i ∈ findall(output.precision .<= 0)
		output.val[i] = 0.5 .* (output.val[i- CartesianIndex(1,0)] + output.val[i+ CartesianIndex(1,0)])
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
    ϕ = atan.((C -A).val , (D-B).val)
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

function  gravi_build_ABCD_phasors(ϕ::AbstractArray{T,2},A,B,C,D) where T
	phasors = zeros(Float64,size(ϕ,1),4,2)
	for l ∈ axes(ϕ,1)
		phi,a,b,c,d = view(ϕ,l,:),view(A,l,:),view(B,l,:),view(C,l,:),view(D,l,:)
		if any(isnan,phi) || any(iszero,phi)
			continue
		end
		(pA,pB,pC,pD) = gravi_estimate_ABCD_phasor(phi,a,b,c,d)
		phasors[l,1,:] .= pA[:]
		phasors[l,2,:] .= pB[:]
		phasors[l,3,:] .= pC[:]
		phasors[l,4,:] .= pD[:]
	end
	return phasors
end

function gravi_build_ABCD_phasors(ϕ::AbstractArray{T,3},A,B,C,D) where {T}
	phasors = zeros(Float64,size(ϕ,1),4,2)
	#trailing = [Colon() for i=2:N]

	for l ∈ axes(ϕ,1)
		phi,a,b,c,d = view(ϕ,l,:,:),view(A,l,:),view(B,l,:),view(C,l,:),view(D,l,:)
		if any(.!isfinite,phi) || any(iszero,phi)
			continue
		end
		(pA,pB,pC,pD) = gravi_estimate_ABCD_phasor(phi,a,b,c,d)
		phasors[l,1,:] .= pA[:]
		phasors[l,2,:] .= pB[:]
		phasors[l,3,:] .= pC[:]
		phasors[l,4,:] .= pD[:]
	end
	return phasors
end

function estimate_visibility(phasors,A,B,C,D;
						robust=false)
	phase =zeros(Float64,size(A)...,2)
	for l ∈ axes(A,1)
		P = phasors[l,:,:]
		#P[:,2] .*=-1
		#if any(x->(iszero(x)||isnan(x)),P[:]) 
		if any(x->(isnan(x)),P) || all(iszero,P)
			continue
		end

		for t ∈ axes(A,2)
			input = [A.val[l,t] ;B.val[l,t] ;C.val[l,t]; D.val[l,t]]
			w = [A.precision[l,t];B.precision[l,t];C.precision[l,t];D.precision[l,t]]
			if sum(iszero.(w))>2 || any(isnan,w[:])
				continue #phase[l,t,:] .=  zeros(Float64,2)
			else
				phase[l,t,:] .= (inv(P'*(w.*P))*P'*(w.*input))[:]
				if robust
					res = sqrt.(w) .* (P * phase[l,t,:] .- input) 
					w .*= (-2.795 .< res .<  2.795)
					if  sum(iszero.(w))>2 || any(isnan,w[:])
						continue
					end
					phase[l,t,:] .= (inv(P'*(w.*P))*P'*(w.*input))[:]
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
			rho = sqrt.(visibilities[:,:,1].^2 .+ visibilities[:,:,2] .^2)
			#rho3 = (ones(360) .* median(rho,dims=1))
			rho3 = (ones(360) .* median(rho[50:200,:],dims=1))
			visibilities .*= 1 ./ rho  .* rho3
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

function wavelength_range(profiles::Dict{String,SpectrumModel{A,B, C}}; baselines=baselines_list,padding=0, λmin=0,λmax=1) where  {A,B,C}
	λstep = minimum([mean(diff(ReducingGravity.get_wavelength(p; bnd=false))) for p ∈ values(profiles)])
	wvmin = 1
	wvmax = 0

	for (i,baseline) ∈ enumerate(baselines)
		T1,T2 = baseline

		wvmin = max(λmin,min(wvmin,minimum([profiles["$T1$T2-$chnl-C"].λbnd[1] for chnl ∈["A","B","C","D"]])))
		wvmax = min(λmax,max(wvmax,maximum([profiles["$T1$T2-$chnl-C"].λbnd[2] for chnl ∈["A","B","C","D"]])))
	end


	usable_wvlngth = zeros(Int,2,6)
	for (i,baseline) ∈ enumerate(baselines)
		T1,T2 = baseline
		baseline_wvlngth = mean(hcat((get_wavelength(profiles["$T1$T2-$chnl-C"]) for chnl ∈["A","B","C","D"])...),dims=2)[:]
		fidx = findfirst(x->isfinite(x) && x>=wvmin,baseline_wvlngth)
		lidx = findlast(x->isfinite(x) && x<=wvmax,baseline_wvlngth)
		usable_wvlngth[:,i] .= [fidx,lidx]

	end

	return range(wvmin - padding * λstep,wvmax  +padding *  λstep; step = λstep), usable_wvlngth
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
	profiles::Dict{String,SpectrumModel{A,B, C}},
	wvidx::Vector{Int}; 
	baselines=baselines_list, kwds...)  where {A,B,C}
	
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
	return v,w
end

function extract_correlated_flux(x::AbstractArray{T,N},tλ, λbaselines,kernel;baselines=baselines_list) where {T,N}
	if N ==1 || size(x,2) ==1
		x = reshape(x,6*2+4,:);
	end
	B = ReducingGravity.Interpolator(tλ,kernel)

	photometric= [InterpolatedSpectrum(x[i,:],B) for i∈1:4]
	interferometric = Vector{Vector{ComplexF64}}(undef,6)
	
	for (i,baseline) ∈ enumerate(baselines)
		T1,T2 = baseline
		tel1 = photometric[T1].(λbaselines[i])
		tel2 = photometric[T2].(λbaselines[i])
		interferometric[i] = (x[4+2*(i-1)+1,:] .+ 1im .* x[4+2*(i-1)+2,:])  ./ sqrt.(max.(0., tel1 .* tel2))./2
	end
	return photometric,interferometric
end