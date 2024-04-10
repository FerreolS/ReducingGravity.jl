
function fitprofile(data::AbstractWeightedData{T,2},bndbx::C; center_degree=4, σ_degree=4, thrsld=0.1) where{T,C<:CartesianIndices}

	fulldata = view(data,bndbx)
	spectra = (sum(fulldata.val .* fulldata.precision,dims=2)./ sum(fulldata.precision,dims=2))[:]
	firstidx = findfirst(x -> x>mean(spectra)*thrsld,spectra)
	lastidx = findlast(x -> x>mean(spectra)*thrsld,spectra)


	data = view(data,bndbx[firstidx:lastidx,:])
	
	specmodel = ProfileModel(bndbx[firstidx:lastidx,:];maxdeg = max(center_degree,σ_degree), precond=true)

	#shp = (sum(data .* wght,dims=1)./ sum(wght,dims=1))[:]


	center = zeros(center_degree+1)
	σ = zeros(σ_degree+1)
	center[1] = mean(bndbx.indices[2])
	
	σ[1] = 0.5 #std((shp .* ay) ./ sum(shp))
	θ = (;center=center, σ = σ)
	params, unflatten = destructure(θ)
	f(params) = likelihood(data,specmodel(;unflatten(params)...))

	res = optimize(f, params, NelderMead(),Optim.Options(iterations=10000))
	xopt = Optim.minimizer(res)
	θopt= unflatten(xopt)
	(;center,σ) = θopt
	center .*=  specmodel.preconditionner[1:σ_degree+1]
	σ .*=  specmodel.preconditionner[1:σ_degree+1]
	θopt = (;center=center,σ=σ)
	return θopt
end


#= 
function test(sm::SpectrumModel{T};center=[0.0],fwhm=[1.0],amplitude=[1.0]) where {T}
	out=Matrix{T}(undef,length(sm.ax), length(sm.ay))
	for (iy,y) ∈ enumerate(sm.ay)
		c,f,amp=  zeros(T,3)
		
		for (center_index,center_coefs) ∈ enumerate(center)
			c += T(sm.preconditionner[center_index]*center_coefs .* y.^ (center_index-1))
		end
		for (fwhm_index,fwhm_coefs) ∈ enumerate(fwhm)
			f += T(sm.preconditionner[fwhm_index]*fwhm_coefs .* y.^ (fwhm_index-1))
		end
		for (amp_index,amp_coefs) ∈ enumerate(amplitude)
			amp += T(sm.preconditionner[amp_index]*amp_coefs .* y.^(amp_index-1))
		end
		halfprecision = inv((T(2/ (2 * sqrt(2 * log(2.))))*f)^2)

		for (ix,x) ∈ enumerate(sm.ax)
			out[ix,iy] = amp * exp(-(x - c)^2 * halfprecision)
		end
	end
	return out
end =#
 