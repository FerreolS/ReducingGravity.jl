@enum Baseline  B12 B13 B23 B41 B42 B43
@enum Chnl  chA chB chC chD

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


function f(data::AbstractWeightedData{T,N}) where {T,N}
	wd = gravi_create_weighteddata(d, illuminated,goodpix,rov, gain, dark = darkp2vm.val)
	ABCD = ConcreteWeightedData{T,N}
	m = gravi_extract_profile(wd,profile)

end

function  gravi_extract_channel(data::AbstractWeightedData,
								profile::SpectrumModel,
								lamp) 
	chnl = gravi_extract_profile(data,profile) 
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
		if any(isnan,phi) #|| any(iszero,phi)
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
		if any(isnan,phi) #|| any(iszero,phi)
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

function estimate_phase(phasors,A,B,C,D;
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