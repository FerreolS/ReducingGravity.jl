function unwrap!(x::AbstractVector{T}; period = 2π) where T
	y = convert(T, period)
	v = first(x)
	@inbounds for k = eachindex(x)
			x[k] = v = v + rem(x[k] - v,  y, RoundNearest)
	end
end

function unwrap(x::AbstractVector{T};  kwds...) where T 
	output = copy(x)
	unwrap!(output; kwds...)
	return output
end


function affine_solve(data::AbstractArray,abscisse::AbstractArray)
	N = length(data)
	length(abscisse) == N || error("data and abscisse does not have the same lenght")
	st = zero(promote_type(eltype(data),eltype(abscisse)))
	st2 = zero(promote_type(eltype(data),eltype(abscisse)))
	sy = zero(promote_type(eltype(data),eltype(abscisse)))
	sty = zero(promote_type(eltype(data),eltype(abscisse)))

    @inbounds @simd for i in eachindex(data, abscisse)
        st  += abscisse[i] 
        st2 += abscisse[i]^2
		sy  += data[i]
		sty += data[i] * abscisse[i] 	
	end
	
	iΔ = 1 / (N*st2 - st^2)
	intercept = iΔ * (st2 * sy - st * sty)
	slope =  iΔ * (N * sty - st * sy )
	return intercept, slope
end


function affine_solve(data::AbstractVector,abscisse::AbstractVector,weights::AbstractVector)
	length(abscisse) == length(data) || error("data and abscisse does not have the same lenght")
	st = zero(promote_type(eltype(data),eltype(abscisse),eltype(weights)))
	st2 = zero(promote_type(eltype(data),eltype(abscisse),eltype(weights)))
	sy = zero(promote_type(eltype(data),eltype(abscisse),eltype(weights)))
	sty = zero(promote_type(eltype(data),eltype(abscisse),eltype(weights)))
	sw = zero(promote_type(eltype(data),eltype(abscisse),eltype(weights)))

    @inbounds @simd for i in eachindex(data, abscisse, weights)
        st  += abscisse[i]  .* weights[i]
        st2 += abscisse[i]^2  .* weights[i]
		sy  += data[i] .* weights[i]
		sty += data[i] * abscisse[i] .* weights[i]
		sw  += weights[i]
	end
	
	iΔ = 1 / (sw * st2 - st^2)
	intercept = iΔ * (st2 * sy - st * sty)
	slope =  iΔ * (sw * sty - st * sy )
	return intercept, slope
end