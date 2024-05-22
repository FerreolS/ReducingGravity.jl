struct WeightedData{T,N,A<:AbstractArray{T,N},B<:AbstractArray{T,N}}# <: AbstractArray{T,N}
	val::A
	precision::B
	function WeightedData(val::A,precision::B) where {T,N,A<:AbstractArray{T,N},B<:AbstractArray{T,N}} 
		size(val) == size(precision) || error("WeightedData : val ≠ precision ")
		new{T,N,A,B}(val,precision)
    end
end
AbstractWeightedData{T,N} = WeightedData{T,N,A,B} where {T,N,A,B}
ConcreteWeightedData{T,N} = WeightedData{T,N,Array{T,N},Array{T,N}} where {T,N}

Base.size(A::WeightedData) = size(A.val)
Base.size(A::WeightedData,n::Int) = size(A.val,n)
Base.length(A::WeightedData) = prod(size(A))
Base.axes(A::WeightedData,n::Int) = axes(A.val,n)

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

Base.:+(A::AbstractWeightedData, B::AbstractWeightedData)  = WeightedData(A.val .+ B.val, 1 ./ ( 1 ./ A.precision .+ 1 ./ B.precision))
Base.:+(A::AbstractWeightedData, B)  = WeightedData(A.val .+ B, A.precision )
Base.:-(A::AbstractWeightedData, B::AbstractWeightedData)  = WeightedData(A.val .- B.val, 1 ./ ( 1 ./ A.precision .+ 1 ./ B.precision))
Base.:-(A::AbstractWeightedData, B)  = WeightedData(A.val .- B, A.precision )
Base.:/(A::AbstractWeightedData, B)  = WeightedData(A.val ./ B, B.^2 .* A.precision)
Base.:*( B, A::AbstractWeightedData)  = WeightedData(A.val .* B,  A.precision ./ B.^2 )
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

function scaledlikelihood(A::D,model::AbstractArray) where {D<:WeightedData}
	α = max.(0,sum(model .* A.precision .* A.val,dims=2) ./ sum( model .*  A.precision .* model,dims=2) )
	
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
	∂Y(Δy) = (NoTangent(),NoTangent(), ZeroTangent())
	return getamplitude(data, model), ∂Y
end

function ChainRulesCore.frule( ::typeof(getamplitude),data::AbstractWeightedData,model)
	∂Y(Δy) = (NoTangent(),NoTangent(), ZeroTangent())
	return getamplitude(data, model), ∂Y
end
