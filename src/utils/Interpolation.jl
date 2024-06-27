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
			s = sum(weights[1:(1-off)])
			weights = weights[(1 - off):end]
			weights = (s,weights[2:end]...)
			off = 0			
		elseif (off+N) > col
			mx = min(off + N, col )
			s = sum(weights[(mx-off):end])
			weights = weights[1:(mx-off)] 
			weights = (weights[1:end-1]...,s)
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
 	


function compute_coefs(I::Interpolator, x,A::AbstractWeightedData; Chi2 =nothing)
		isnothing(Chi2) && return compute_coefs(I, x,A.val,A.precision)
		return compute_coefs(I, x,A,Chi2)
	end

function compute_coefs((;kernel, knots)::Interpolator, x,A::AbstractWeightedData,Chi2::Float64)
	(;val, precision) = A
	N = sum(precision .>0)
	K = build_interpolation_matrix(kernel,knots,x)
	R = build_interpolation_matrix(kernel',knots,x)
	KK = Hermitian(K' * (precision .* K))
	RR = Hermitian(R'*R)
	function f(μ)
		C = Hermitian( KK .+ (10.0.^μ) .* RR)
		F = cholesky(C; check=false)
		if issuccess(F)
			out =   F \ K' * (precision .* val)
			return likelihood(A,K*out) ./ N - Chi2
		else
			out = Symmetric(pinv(C)) *  K' * (precision .* val)
			return likelihood(A,K*out) ./ N - Chi2
		end
	end
	a = -9.
	fa=f(a)
	if fa > 0
		F = cholesky(KK; check=false)
		if issuccess(F)
			return F \ K' * (precision .* val)
		else
			return Symmetric(pinv(KK)) *  K' * (precision .* val)
		end
	end
	b= 1.
	fb = f(b)
	while fb < 0
		a=b
		fa=fb
		b += 1
		fb = f(b)
	end	
	(μ, f1, lo1, hi1, n1)  = OptimPackNextGen.Brent.fzero(f,a,fa,b,fb)
	@debug	μ, f1, lo1, hi1, n1
	C = Hermitian( KK .+ (10.0.^μ).* RR)
	F = cholesky(C; check=false)
	if issuccess(F)
		return F \ K' * (precision .* val)
	else
		return Symmetric(pinv(C)) *  K' * (precision .* val)
	end
end
