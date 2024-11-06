struct Interpolator{A,B}
	knots::A
	kernel::B
end


function build_interpolation_matrix(kernel::Kernel{T,N}, knots, samples) where {T,N}
	lin = length(samples)
	col = length(knots) 
	K = zeros(T,lin,col)
 	for (l,sample) ∈ enumerate(samples)
		offweights = InterpolationKernels.compute_offset_and_weights(kernel,T.(find_index(knots,sample))) 
		weights = vcat(offweights[2]...)
		off::Int = round(Int,offweights[1]) +1

		if off <= 0 
			if (off+N)<=0 
				continue
			end
			weights = weights[(1 - off):end]
			off = 1			
			weights = (sw=sum(weights))==0 ? weights : weights./sw
		elseif (off+N) > col
			if off>col 
				continue
			end

			weights = weights[1:(col-off+1)] 
			weights = (sw=sum(weights))==0 ? weights : weights./sw
		end
		wsz = length(weights)
		K[l,off:(off+wsz-1)] .= weights
	end
	return K
end

build_interpolation_matrix((;knots,kernel)::Interpolator, samples) = build_interpolation_matrix(kernel, knots, samples) 

function find_index(knots::StepRangeLen,sample)
	return (sample  - first(knots)) /step(knots) +1
end

function compute_coefs((;kernel, knots)::Interpolator, x,y)
	K = build_interpolation_matrix(kernel,knots,x)
	C = Symmetric(K'*K)
    F = cholesky(C; check=false)
    if issuccess(F)
        return   F \ K' * y
    else
        return Symmetric(pinv(C)) * K' * y
    end
end

function compute_coefs((;kernel, knots)::Interpolator, 
						x,
						y::AbstractArray{T,N},
						w::AbstractArray{T,N}) where {T,N}
						
	K = build_interpolation_matrix(kernel,knots,x)
	return compute_coefs(K,y,w)

end

function compute_coefs(K::AbstractMatrix,
						y::AbstractVector,
						w::AbstractVector) 
	C = Symmetric(K' * (w .* K) )
    F = cholesky(C; check=false)
    if issuccess(F)
        return   F \ K' * (w .* y)
    else
        return Symmetric(pinv(C)) *  K' * (w .* y)
    end			
end


function compute_coefs(K::AbstractMatrix,
						y::AbstractMatrix,
						w::AbstractMatrix) 
						
	out = similar(y,(size(K,2),size(y,2)))
	@inbounds Threads.@threads for i ∈ axes(y,2)
		out[:,i] .= compute_coefs(K,y[:,i],w[:,i])
	end	
#= 
	Threads.@threads for (oi,yi,wi) ∈ zip(eachcol(out),eachcol(y),eachcol(w))
		oi .= compute_coefs(K,yi,wi)
	end	 =#
	return out	
end

compute_coefs(K::AbstractMatrix,(;val,precision)::AbstractWeightedData) = compute_coefs(K,val, precision)


function compute_coefs(I::Interpolator, x,A::AbstractWeightedData; Chi2 =nothing)
		isnothing(Chi2) && return compute_coefs(I, x,A.val,A.precision)
		return compute_coefs(I, x,A,Chi2)
end

function compute_coefs((;kernel, knots)::Interpolator, x,A::AbstractWeightedData,Chi2::Float64) 
	T = eltype(kernel)
	(;val, precision) = A
	N = sum(precision .>0)
	K = build_interpolation_matrix(kernel,knots,x)
	compute_coefs(K,A,Chi2)
end


function compute_coefs(K::AbstractMatrix{T},A::AbstractWeightedData,Chi2::Float64)  where T
	(;val, precision) = A
	N = sum(precision .>0)
	#R = build_interpolation_matrix(kernel',knots,x)
	KK = Symmetric(K' * (precision .* K))
	RR = make_DtD(T,size(KK,1))
	B = K' * (precision .* val)
	function f(μ)
		C = Symmetric( KK .+ T(10.0.^μ) .* RR)
		F = cholesky(C; check=false)
		if issuccess(F)
			out =   F \ B # (K' * (precision .* val))
			return likelihood(A,K*out) ./ N - Chi2
		else
			out = Symmetric(pinv(C)) * B
			return likelihood(A,K*out) ./ N - Chi2
		end
	end
	a = -9.
	fa=f(a)
	if fa > 0
		F = cholesky(KK; check=false)
		if issuccess(F)
			return F \B
		else
			return Symmetric(pinv(KK)) *  B
		end
	end
	b=-8.
	fb = f(b)
	while fb < 0
		a=b
		fa=fb
		b += 1
		fb = f(b)
	end	
	@debug a,fa,b,fb
	(μ, f1, lo1, hi1, n1)  = OptimPackNextGen.Brent.fzero(f,a,fa,b,fb)
	@debug	μ, f1, lo1, hi1, n1
	C = Symmetric( KK .+ T(10.0.^μ).* RR)
	F = cholesky(C; check=false)
	if issuccess(F)
		return F \ B
	else
		return Symmetric(pinv(C)) *  B
	end
end


function make_DtD(T::DataType,n)
	diag0 = Vector{T}(undef,n)
	fill!(diag0,T(2))
	diag0[1] = T(1)
	diag0[end] = T(1)
	diag1 = Vector{T}(undef,n-1)
	fill!(diag1,T(-1))
	V = vcat(diag0,diag1,diag1)
	I = vcat(1:n,2:n,1:(n-1))
	J = vcat(1:n,1:(n-1),2:n)
	Symmetric(sparse(I, J, V))
end