
struct Interpolator{A,B}
	knots::A
	kernel::B
end


function build_interpolation_matrix(kernel::Kernel{T,N}, knots, samples) where {T,N}
	lin = length(samples)
	col = length(knots) 
	K = zeros(T,lin,col)
 	for (l,sample) ∈ enumerate(samples)
		off, weights = InterpolationKernels.compute_offset_and_weights(kernel,find_index(knots,sample)) 
		off = Int(off)
		mx = off + N
		if off < 0 
			weights = weights[(1 - off):end]
			off = 0			
			weights = (sw=sum(weights))==0 ? weights : weights./sw
		elseif (off+N) > col
			mx = min(off + N, col )
			weights = weights[1:(mx-off)] 
			weights = (sw=sum(weights))==0 ? weights : weights./sw
		end
		K[l,(off+1):mx] .= weights
	end
	return K
end

build_interpolation_matrix((;knots,kernel)::Interpolator, samples) = build_interpolation_matrix(kernel, knots, samples) 
#= function find_index(knots,sample)
	b = findfirst(sample .<= knots)
	b = isnothing(b) ? firstindex(knots) : b 
	kb = knots[b]
	kb ==sample && return kb
	e = findlast(sample .> knots)
	e = isnothing(e) ? firstindex(knots) : e
	ke = knots[e]
	ke == sample && return ke
	b == e && return ke
	return b + (sample - kb) / (ke - kb) 
end =#

function find_index(knots::StepRangeLen,sample)
	return (sample  - first(knots)) /step(knots) 
end

function compute_coefs((;kernel, knots)::Interpolator, x,y)
	K = build_interpolation_matrix(kernel,knots,x)
	C = Hermitian(K'*K)
    F = cholesky(C; check=false)
    if issuccess(F)
        return   F \ K' * y
    else
        return Symmetric(pinv(C)) * K' * y
    end
end

function compute_coefs((;kernel, knots)::Interpolator, x,y,w)
	K = build_interpolation_matrix(kernel,knots,x)
	C = Hermitian(K' * (w .* K))
    F = cholesky(C; check=false)
    if issuccess(F)
        return   F \ K' * (w .* y)
    else
        return Symmetric(pinv(C)) *  K' * (w .* y)
    end
end

function compute_coefs((;kernel, knots)::Interpolator, x,(;val, precision)::AbstractWeightedData)
	K = build_interpolation_matrix(kernel,knots,x)
	C = Hermitian(K' * (precision .* K))
    F = cholesky(C; check=false)
    if issuccess(F)
        return   F \ K' * (precision .* val)
    else
        return Symmetric(pinv(C)) *  K' * (precision .* val)
    end
end