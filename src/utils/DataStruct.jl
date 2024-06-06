



struct InterpolatedSpectrum{B}
	coefs::Vector{Float64}
	basis::B
end

#(self::InterpolatedSpectrum{B})(x) where B<:BSplineBasis = Spline(self.basis,self.coefs)(x)
#(self::InterpolatedSpectrum{B})() where B<:BSplineBasis = Spline(self.basis,self.coefs)

function (self::InterpolatedSpectrum{B})(x) where B<:Interpolator
	(;knots, kernel) = self.basis
	notnan = isfinite.(x)
	if any(notnan)
		basis = build_interpolation_matrix(kernel,knots,view(x, notnan))
		out = copy(x)
		view(out,notnan) .= basis*self.coefs
		return out
	else
		basis = build_interpolation_matrix(kernel,knots,x)
		return basis*self.coefs
	end
end
function (self::InterpolatedSpectrum{B})(x::Number) where B<:Interpolator
	!isfinite(x) && return x 
	(;knots, kernel) = self.basis
	x = min(x,knots[end])
	x = max(x,knots[1])
	basis = build_interpolation_matrix(kernel,knots,x)
	return (basis*self.coefs)[1]
end
#= 
function (self::InterpolatedSpectrum{B})() where B<:Interpolator
	basis = build_interpolation_matrix(kernel,knots,x)
	return self.basis.basis * self.coefs
end
function recompute_basis(S::Interpolator,x)
	(;knots, kernel, _) = S.basis
	basis = build_interpolation_matrix(kernel,knots,x)
	return Interpolator(knots, kernel, basis)
end
 =#


struct SpectrumModel{A,B,C,D}
	center::Vector{Float64}
	σ::Vector{Float64}
	λ::B
	λbnd::Vector{Float64}
	transmissions::Vector{InterpolatedSpectrum{C}}
	flat::D
	bbox::A
end

function ((;center,σ)::SpectrumModel{A,Nothing,B,D})(p) where {A,B,D}
	cdeg = length(center)
	cp =  p .^(0:(cdeg-1))'* center
	σdeg = length(σ)
	σp = p .^(0:(σdeg-1))'* σ
	#λdeg = length(λ)
 	#λp = p .^(0:(λdeg-1))'* λ
	return (;center=cp[1],σ=σp[1])#,λ=λp[1])
end

function get_center(s::SpectrumModel{A,B,C,D}) where {A,B,C,D}
	(;center,bbox) = s 
 	if C == Nothing
		p = bbox.indices[1]
	else
		p = get_wavelength(s)
	end
	cdeg = length(center)	
	return p .^(0:(cdeg-1))'* center
end

function get_width((;σ,bbox)::SpectrumModel)
	p = bbox.indices[1]
	σdeg = length(σ)	
	return p .^(0:(σdeg-1))'* σ
end


get_wavelength(::SpectrumModel{A,Nothing,B,D},kwds...) where {A,B,D} = nothing

function get_wavelength((;λ)::SpectrumModel,p;kwds...)
	λdeg = length(λ)
 	λp = p .^(0:(λdeg-1))'* λ
	return λp[1]
end

function get_wavelength((;λ,λbnd, bbox)::SpectrumModel; bnd=true)
	p = bbox.indices[1]
	λdeg = length(λ)	
	wv = p .^(0:(λdeg-1))'* λ
	if bnd
		wv[ .!(λbnd[1] .< wv .<λbnd[2])] .= NaN
	end

	return wv
end


struct ProfileModel{A1,P} 
	bbox::A1
	preconditionner::P
end

function ProfileModel(bbox::A1;maxdeg=3, precond=false) where {A1}
	
	if precond 
		ax = bbox.indices[1]
		preconditionner  = [ sqrt(length(ax) /sum(Float64.(ax).^(2*n)) ) for n ∈ 0:maxdeg]
	else
		preconditionner  = nothing
	end
    ProfileModel{A1,typeof(preconditionner)}(bbox,preconditionner)
end 

function (self::ProfileModel{A1,P})(;center=[0.0],σ=[1.0],amplitude=[1.0]) where {A1,P}
	ncenter = length(center)
	nσ = length(σ)
	namp = length(amplitude)
	ax = self.bbox.indices[1]
	ay = self.bbox.indices[2]

	degmax = max(ncenter,nσ,namp)
	if P == Nothing
		u = broadcast(^,ax,(0:(degmax-1))')
	else
		u = broadcast(^,ax,(0:(degmax-1))').* self.preconditionner[1:degmax]'
	end
	cy = sum(u[:,1:ncenter].*center',dims=2)
	ampy = sum(u[:,1:namp].*amplitude',dims=2)

	sy = sum(u[:,1:nσ].*σ',dims=2)

	return ampy .* exp.(-1 ./ 2 .*((cy .- ay')./ sy).^2)
end

(self::ProfileModel)((;center,σ)::SpectrumModel{A,Nothing,B,D}) where {A,B,D} = self(;center=center, σ=σ)

function get_profile(profile::SpectrumModel) 
	(;center,σ) = profile
	ncenter = length(center)
	nσ = length(σ)
	ay = profile.bbox.indices[2]

	degmax = max(ncenter,nσ)

    λ = get_wavelength(profile)
	u = broadcast(^,λ,(0:(degmax-1))')

	cy = sum(u[:,1:ncenter].*center',dims=2)
	sy = sum(u[:,1:nσ].*σ',dims=2)
	return exp.(-1 ./ 2 .*((cy .- ay')./ sy).^2)
end


function get_profile(s::SpectrumModel{A,Nothing,B,D}) where {A,B,D}
	ProfileModel(s.bbox)(;s.center,s.σ)
end

function get_profile(s::SpectrumModel{A,Nothing,B,D},bndbox) where {A,B,D}
	ProfileModel(bndbox)(;s.center,s.σ)
end