struct WeightedData{T,N,A<:AbstractArray{T,N},B<:AbstractArray{T,N}}# <: AbstractArray{T,N}
	val::A
	precision::B
	function WeightedData(val::A,precision::B) where {T,N,A<:AbstractArray{T,N},B<:AbstractArray{T,N}} 
		size(val) == size(precision) || error("WeightedData : val ≠ precision ")
		new{T,N,A,B}(val,precision)
    end
end
AbstractWeightedData{T,N} = WeightedData{T,N,A,B} where {T,N,A,B}

Base.size(A::WeightedData) = size(A.val)
Base.length(A::WeightedData) = prod(size(A))

Base.getindex(A::WeightedData, I::Vararg{Int, N}) where N	= (;val=A.val[I],precision=A.precision[I])
Base.getindex(A::WeightedData, I::Int)	= (;val=A.val[I],precision=A.precision[I])
Base.getindex(A::WeightedData, I...)	= (;val=A.val[I],precision=A.precision[I])
function Base.setindex!(A, (;val,precision), I)
    setindex!(A.val, val, I)
    setindex!(A.precision, precision, I)
end


function Base.view(A::WeightedData{T,N}, I...) where {T,N}
	WeightedData(view(A.val,I...),view(A.precision,I...))
end

Base.:+(A::AbstractWeightedData{T,N}, B::AbstractWeightedData{T,N}) where {T,N} = WeightedData(A.val .+ B.val, 1 ./ ( 1 ./ A.precision .+ 1 ./ B.precision))
Base.:-(A::AbstractWeightedData{T,N}, B::AbstractWeightedData{T,N}) where {T,N} = WeightedData(A.val .- B.val, 1 ./ ( 1 ./ A.precision .+ 1 ./ B.precision))
Base.:-(A::AbstractWeightedData{T,N}, B::AbstractArray{T,N}) where {T,N} = WeightedData(A.val .- B, A.precision )
Base.:/(A::AbstractWeightedData, B::Number)  = WeightedData(A.val ./ B, B.^2 .* A.precision)
Base.:*( B::Number, A::AbstractWeightedData)  = WeightedData(A.val .* B,  A.precision ./ B.^2 )
Base.:*(A::AbstractWeightedData, B::Number)  = B * A

function flagbadpix!(A::WeightedData{T,N},badpix::Union{ Array{Bool, N},BitArray{N}}) where {T,N}
    A.val[badpix] .= T(0) 
	A.precision[badpix] .= T(0)
end

function likelihood(A::D,model::AbstractArray) where {D<:WeightedData}
	return sum( (A.val .- model).^2 .* A.precision)/ 2
end 

function ChainRulesCore.rrule( ::typeof(likelihood),A::D,model::AbstractArray) where {D<:WeightedData}
	r =(model .- A.val)
	rp = r .* A.precision
    likelihood_pullback(Δy) = (NoTangent(),NoTangent(), rp .* Δy)
    return  sum(r.*rp) / 2, likelihood_pullback
end


struct Profile{A}
	center::Vector{Float64}
	σ::Vector{Float64}
	λ::Vector{Float64}
	bbox::A
end

function (self::Profile)(p)
	(;center,σ,λ) = self
	cdeg = length(center)
	cp =  p .^(0:(cdeg-1))'* center
	σdeg = length(σ)
	σp = p .^(0:(σdeg-1))'* σ
	λdeg = length(λ)
 	λp = p .^(0:(λdeg-1))'* λ
	return (;center=cp,σ=σp,λ=λp)
end

struct Transmission{T,B}
	coefs::Vector{T}
	SplineBasis::B
end

(self::Transmission)(x) = Spline(self.SplineBasis,self.coefs)(x)
(self::Transmission)() = Spline(self.SplineBasis,self.coefs)
