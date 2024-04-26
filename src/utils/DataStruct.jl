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

Base.:+(A::AbstractWeightedData{T,N}, B::AbstractWeightedData{T,N}) where {T,N} = WeightedData(A.val .+ B.val, 1 ./ ( 1 ./ A.precision .+ 1 ./ B.precision))
Base.:-(A::AbstractWeightedData{T,N}, B::AbstractWeightedData{T,N}) where {T,N} = WeightedData(A.val .- B.val, 1 ./ ( 1 ./ A.precision .+ 1 ./ B.precision))
Base.:-(A::AbstractWeightedData{T,N}, B::AbstractArray{T,N}) where {T,N} = WeightedData(A.val .- B, A.precision )
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

struct Transmission{B}
	coefs::Vector{Float64}
	SplineBasis::B
end

(self::Transmission)(x) = Spline(self.SplineBasis,self.coefs)(x)
(self::Transmission)() = Spline(self.SplineBasis,self.coefs)



struct SpectrumModel{A,B}
	center::Vector{Float64}
	σ::Vector{Float64}
	λ::Vector{Float64}
	transmissions::Vector{Transmission{B}}
	bbox::A
end

function (self::SpectrumModel)(p)
	(;center,σ,λ) = self
	cdeg = length(center)
	cp =  p .^(0:(cdeg-1))'* center
	σdeg = length(σ)
	σp = p .^(0:(σdeg-1))'* σ
	λdeg = length(λ)
 	λp = p .^(0:(λdeg-1))'* λ
	return (;center=cp[1],σ=σp[1],λ=λp[1])
end

function get_profile(s::SpectrumModel)
	ProfileModel(s.bbox)(;s.center,s.σ)
end

function get_profile(s::SpectrumModel,bndbox)
	ProfileModel(bndbox)(;s.center,s.σ)
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

(self::ProfileModel)((;center,σ)::SpectrumModel) = self(;center=center, σ=σ)
