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

	for (key,profile) ∈ profiles
		if (tel1 == key[1]) && (tel2 == key[2])
			continue
		end
		key1 = "$tel1-$key" 
		key2 = "$tel2-$key" 
	end
	for chnl ∈ ["A","B","C","D"]
		profile["$tel1$tel2-$chnl-C"]
	end	
end