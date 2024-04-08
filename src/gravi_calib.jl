struct SpectrumModel{T,A1,P} 
	bbox::A1
	preconditionner::P
end

function SpectrumModel{T}(bbox::A1;maxdeg=3, precond=false) where {T,A1}
	
	if precond 
		ax = bbox.indices[1]
		preconditionner  = [ sqrt(length(ax) /sum(Float64.(ax).^(2*n)) ) for n ∈ 0:maxdeg]
	else
		preconditionner  = 1
	end
    SpectrumModel{T,A1,typeof(preconditionner)}(bbox,preconditionner)
end 

function (self::SpectrumModel{T,A1,P})(;center=[0.0],σ=[1.0],amplitude=[1.0]) where {T,A1,P}
	ncenter = length(center)
	nσ = length(σ)
	namp = length(amplitude)
	ax = self.bbox.indices[1]
	ay = self.bbox.indices[2]

	degmax = max(ncenter,nσ,namp)
	if P <:Number
		u = broadcast(^,ax,(0:(degmax-1))')
	else
		u = broadcast(^,ax,(0:(degmax-1))').* self.preconditionner[1:degmax]'
	end
	cy = sum(u[:,1:ncenter].*center',dims=2)
	ampy = sum(u[:,1:namp].*amplitude',dims=2)

	sy = sum(u[:,1:nσ].*σ',dims=2)

	return T.(ampy .* exp.(-1 ./ 2 .*((cy .- ay')./ sy).^2))
end
(self::SpectrumModel)((;center,σ)::Profile) = self(;center=center, σ=σ)

function scaledweightedloss(model,data, weights)
	α = max.(0,sum(model .* weights .* data,dims=2) ./ sum( model .* weights .* model,dims=2) )
	res = ( α .* model .- data) 
	return sum(res.^2 .* weights)
end

function fitprofile(data::AbstractMatrix{T},wght::AbstractMatrix{T},bndbx::C; center_degree=4, σ_degree=4, thrsld=0.1) where{T,C<:CartesianIndices}

	spectra = (sum(data .* wght,dims=2)./ sum(wght,dims=2))[:]
	firstidx = findfirst(x -> x>mean(spectra)*thrsld,spectra)
	lastidx = findlast(x -> x>mean(spectra)*thrsld,spectra)


	data = view(data,firstidx:lastidx,:)
	wght = view(wght,firstidx:lastidx,:)
	
	specmodel = SpectrumModel{T}(bndbx[firstidx:lastidx,:];maxdeg = max(center_degree,σ_degree), precond=true)

	#shp = (sum(data .* wght,dims=1)./ sum(wght,dims=1))[:]


	center = zeros(center_degree+1)
	σ = zeros(σ_degree+1)
	center[1] = mean(bndbx.indices[2])
	
	σ[1] = 0.5 #std((shp .* ay) ./ sum(shp))
	θ = (;center=center, σ = σ)
	params, unflatten = destructure(θ)
	f(params) = scaledweightedloss(specmodel(;unflatten(params)...),data, wght)

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
 