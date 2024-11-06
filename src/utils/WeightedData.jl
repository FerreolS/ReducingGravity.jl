struct WeightedData{T,N,A<:AbstractArray{T,N},B<:AbstractArray{T,N}}# <: AbstractArray{T,N}
	val::A
	precision::B
	function WeightedData(val::A,precision::B) where {T,N,A<:AbstractArray{T,N},B<:AbstractArray{T,N}} 
		size(val) == size(precision) || error("WeightedData : val ≠ precision ")
		new{T,N,A,B}(val,precision)
    end
end
WeightedData(val::A,precision::Number) where {A<:Number} = WeightedData(vcat(val),vcat(A(precision)))
WeightedData((;val, precision)::WeightedData) = WeightedData(val,precision)
WeightedData((;val, precision)::WeightedData,I...) = WeightedData(val[I...],precision[I...])


AbstractWeightedData{T,N} = WeightedData{T,N,A,B} where {T,N,A,B}
ConcreteWeightedData{T,N} = WeightedData{T,N,Array{T,N},Array{T,N}} where {T,N}

Base.size(A::WeightedData) = size(A.val)
Base.size(A::WeightedData,n::Int) = size(A.val,n)
Base.length(A::WeightedData) = prod(size(A))
Base.axes(A::WeightedData,n::Int) = axes(A.val,n)

#Base.getindex(A::WeightedData, I::Vararg{Int, N}) where N	= WeightedData(A.val[I],A.precision[I])
#Base.getindex(A::WeightedData, I::Int)	= WeightedData(A.val[I],A.precision[I])
Base.getindex(A::WeightedData, I...)	= WeightedData(A, I...)
function Base.setindex!(A::WeightedData, (;val,precision), I)
    setindex!(A.val, val, I)
    setindex!(A.precision, precision, I)
end


function Base.view(A::WeightedData, I...) 
	WeightedData(view(A.val,I...),view(A.precision,I...))
end

Base.:+(A::AbstractWeightedData, B::AbstractWeightedData)  = WeightedData(A.val .+ B.val, 1 ./ ( 1 ./ A.precision .+ 1 ./ B.precision))
Base.:+(A::AbstractWeightedData, B)  = WeightedData(A.val .+ B, A.precision )
Base.:-(A::AbstractWeightedData, B::AbstractWeightedData)  = WeightedData(A.val .- B.val, 1 ./ ( 1 ./ A.precision .+ 1 ./ B.precision))
Base.:-(A::AbstractWeightedData, B)  = WeightedData(A.val .- B, A.precision )
Base.:/(A::AbstractWeightedData, B)  = WeightedData(A.val ./ B, B.^2 .* A.precision)
Base.:*( B, A::AbstractWeightedData)  = WeightedData(A.val .* B,  A.precision ./ B.^2 )
Base.:*(A::AbstractWeightedData, B)  = B * A

Base.real(A::AbstractWeightedData) = WeightedData(real.(A.val),real.(A.precision))
Base.imag(A::AbstractWeightedData) = WeightedData(real.(A.val),real.(A.precision))

combine(B::NTuple{N,W}) where {N,W <: AbstractWeightedData}  = combine(first(B),last(B, N-1)...)
combine(A::AbstractWeightedData, B...)   = combine(combine(A,first(B)),last(B, length(B)-1)...)
combine(B::AbstractArray{W}) where W <: AbstractWeightedData  = combine(first(B),last(B, length(B)-1)...)
combine(A::AbstractWeightedData, B::AbstractArray{W}) where W <: AbstractWeightedData  = combine(combine(A,first(B)),last(B, length(B)-1)...)
combine(A::AbstractWeightedData) = A
function combine(A::AbstractWeightedData{T,N}, B::AbstractWeightedData{T,N}) where {T,N}
	precision = A.precision .+B.precision
	val =(A.precision .* A.val .+ B.precision .* B.val)./(precision)
	view(val, iszero.(precision)) .= zero(T) 
    WeightedData(val,precision )
end



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

function scaledlikelihood(A::D,model::AbstractArray{T,N}) where {D<:WeightedData,T,N}
	α = max.(0,sum(model .* A.precision .* A.val,dims=2) ./ sum( model .*  A.precision .* model,dims=2) )
	α[.!isfinite.(α)] .= T(0)
	res = ( α .* model .- A.val) 
	return sum(res.^2 .* A.precision)/2
end
 
function ChainRulesCore.rrule( ::typeof(scaledlikelihood),A::D,model::AbstractArray) where {D<:WeightedData}
	α = max.(0,sum(model .* A.precision .* A.val,dims=2) ./ sum( model .*  A.precision .* model,dims=2) )
	r =( α .*model .- A.val)
	rp = r .* A.precision
    likelihood_pullback(Δy) = (NoTangent(),NoTangent(), α .* rp .* Δy)
    return  sum(r.*rp) / 2, likelihood_pullback
end




function getamplitude(data::AbstractWeightedData,model)
	#return max.(0, ldiv!(cholesky!(Symmetric(model' * ( data.precision.* model))),model'* (data.precision .* (data.val ))))
	return max.(0,pinv(model' * ( data.precision.* model))*model'* (data.precision .* (data.val )))
end
function ChainRulesCore.rrule( ::typeof(getamplitude),data::AbstractWeightedData,model)
	∂Y(_) = (NoTangent(),NoTangent(), ZeroTangent())
	return getamplitude(data, model), ∂Y
end

function ChainRulesCore.frule( ::typeof(getamplitude),data::AbstractWeightedData,model)
	∂Y(_) = (NoTangent(),NoTangent(), ZeroTangent())
	return getamplitude(data, model), ∂Y
end
