"""
    InterpolatedSpectrum{T,B}

A structure representing an interpolated spectrum.

# Fields
- `coefs::Vector{T}`: Interpolation coefficients
- `basis::B`: Interpolation basis of type `Interpolator`

# Description
This structure allows representing and evaluating an interpolated spectrum from
a set of coefficients and an interpolation basis.
"""
struct InterpolatedSpectrum{T,B}
    coefs::Vector{T}
    basis::B
end

"""
    (self::InterpolatedSpectrum)(x::AbstractVector{T2}) where {T,B<:Interpolator,T2<:Number}

Evaluates the interpolated spectrum at a vector of points.

# Arguments
- `x::AbstractVector{T2}`: Evaluation points

# Returns
- Vector of interpolated values at the given points

# Notes
- Non-finite values (NaN, Inf) in `x` are preserved in the output
- Interpolation is performed only on finite values using the basis matrix
"""
function (self::InterpolatedSpectrum{T,B})(x::AbstractVector{T2}) where {T,B<:Interpolator,T2<:Number}
	(;knots, kernel) = self.basis
	notnan = isfinite.(x)
	if any(notnan)
		basis = build_interpolation_matrix(kernel,knots,view(x, notnan))
		out = collect(x)
		view(out,notnan) .= basis*self.coefs
		return out
	else
		basis = build_interpolation_matrix(kernel,knots,x)
		return basis*self.coefs
	end
end

"""
    (self::InterpolatedSpectrum)(x::T2) where {T,B<:Interpolator,T2<:Number}

Evaluates the interpolated spectrum at a single point.

# Arguments
- `x::T2`: Evaluation point

# Returns
- Interpolated value at the given point

# Notes
- If `x` is not finite (NaN, Inf), returns `x` converted to type `T`
- The value of `x` is clamped to the interval defined by the basis knots
"""
function (self::InterpolatedSpectrum{T,B})(x::T2) where {T,B<:Interpolator,T2<:Number}
	!isfinite(x) && return T(x) 
	(;knots, kernel) = self.basis
	x = min(x,knots[end])
	x = max(x,knots[1])
	basis = build_interpolation_matrix(kernel,knots,x)
	return (basis*self.coefs)[1]
end


struct SpectrumModel{A,B,C,T}
	center::Vector{Float64}
	σ::Matrix{Float64}
	λ::B
	λbnd::Vector{Float64}
	transmissions::Vector{InterpolatedSpectrum{T,C}}
	bbox::A
end

function ((;center,σ)::SpectrumModel{A,Nothing,B,E})(p) where {A,B,E}
	cdeg = length(center)
	cp =  p .^(0:(cdeg-1))'* center
	σdeg = length(σ)
	σp = p .^(0:(σdeg-1))'* σ
	return (;center=cp[1],σ=σp)#,λ=λp[1])
end

function get_center(s::SpectrumModel{A,B,C,E}) where {A,B,C,E}
	(;center,bbox) = s 
 	if B == Nothing
		p = bbox.indices[1]
	else
		p = get_wavelength(s)
	end
	cdeg = length(center)	
	return p .^(0:(cdeg-1))'* center
end

function get_width(s::SpectrumModel{A,B,C,E}) where {A,B,C,E}
	(;σ,bbox) = s 
 	if B == Nothing
		p = bbox.indices[1]
	else
		p = get_wavelength(s)
	end
	σdeg = size(σ,1)	
	
	return p .^(0:(σdeg-1))'* σ
end


get_wavelength(::SpectrumModel{A,Nothing,B,E},kwds...) where {A,B,E} = nothing

function get_wavelength((;λ)::SpectrumModel,p::T;kwds...) where T
	λdeg = length(λ)
	if λdeg < 5
 		λp = p .^(0:(λdeg-1))'* λ
		return λp[1]
	else
		if T <:Integer
			return λ[p]
		else
			error("p must be an integer")

		end
	end
end

function get_wavelength((;λ,λbnd, bbox)::SpectrumModel; bnd=false)
	p = bbox.indices[1]
	λdeg = length(λ)	

	if λdeg < 5 
		wv = p .^(0:(λdeg-1))'* λ
	else
		wv = λ
	end
	if bnd
			wv= wv[ (λbnd[1] .<= wv .<=λbnd[2])]
	else
		wv[ .!(λbnd[1] .<= wv .<=λbnd[2])] .= NaN
	end

	return wv
end

get_wavelength_bounds_inpixels((;bbox)::SpectrumModel{A,Nothing,C,E}) where {A,C,E} = bbox.indices[1]

function get_wavelength_bounds_inpixels((;λ,λbnd, bbox)::SpectrumModel{A,B,C,E}) where {A,B,C,E}
	p = bbox.indices[1]
	λdeg = length(λ)	
	if λdeg <5
		wv = p .^(0:(λdeg-1))'* λ
	else
		wv =λ
	end
	return findfirst(x->x>=λbnd[1],wv):findlast(x->x<=λbnd[2],wv)
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
	nσ = size(σ,1)
	namp = length(amplitude)
	ax = self.bbox.indices[1]
	ay = self.bbox.indices[2]

	degmax = max(ncenter,nσ,namp)
	if P == Nothing
		u = broadcast(^,Float64.(ax),(0:(degmax-1))')
	else
		u = broadcast(^,Float64.(ax),(0:(degmax-1))').* self.preconditionner[1:degmax]'
	end
	cy = u[:,1:ncenter]*center
	if namp!=1 && namp[1]!=0.0
		ampy = u[:,1:namp]*amplitude
	else
		ampy = 1.0
	end
	sy = u[:,1:nσ]*σ
	if size(σ,2)==2
		ly = length(ay)
		sy = sy*vcat(range(0,1,ly)',range(1,0,ly)')
	end
	
	return ampy .* exp.(-1 ./ 2 .*((cy .- ay')./ sy).^2)
end

(self::ProfileModel)((;center,σ)::SpectrumModel{A,Nothing,B,E}) where {A,B,E} = self(;center=center, σ=σ)

function get_profile(profile::SpectrumModel) 
		bbox = CartesianIndices((get_wavelength_bounds_inpixels(profile),profile.bbox.indices[2]))
	return ProfileModel(bbox)(;profile.center,profile.σ)
end


function get_profile(s::SpectrumModel{A,Nothing,B,E}) where {A,B,E}
	ProfileModel(s.bbox)(;s.center,s.σ)
end

function get_profile(s::SpectrumModel{A,Nothing,B,E},bndbox) where {A,B,E}
	ProfileModel(bndbox)(;s.center,s.σ)
end