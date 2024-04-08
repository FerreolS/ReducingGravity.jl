# argon line and respective coarse pixel index in Med mode
argon = @SMatrix [ 1.982291e-6 43 
       1.997118e-6 49   
       2.032256e-6 65   
       2.062186e-6 79   
       #2.065277e-6 80   # doublet?
       2.073922e-6 84    
       2.081672e-6 88   
       2.099184e-6 96   
       2.133871e-6 112   
       2.154009e-6 121   
       2.20456e-6 144  #  doublet 
       2.208321e-6 145   
       2.313952e-6 194   
       2.385154e-6 226   
       2.397306e-6 232   ]

function gaussian_lines(rng;center=[0.0],σ=[1.0],amplitude=[1.0]) 
	return amplitude' .* exp.(-1 ./ 2 .*((center' .- rng)./ σ').^2)
end

function argonmodel(rng, data, B; center=[0.0],σ=[1.0], coefs = nothing) 
	(;val, precision)=data

	if isnothing(coefs)
		G = gaussian_lines(rng;center=center,σ=σ)
	else
		G = hcat(gaussian_lines(rng;center=center,σ=σ), (Spline(B,coefs)).(rng) ) #baseline = (Spline(B,coefs)).(rng) 
	end

	#G = gaussian_lines(rng;center=center,σ=σ)
	amplitude = max.(0,pinv(G' * ( precision.* G))*G'* (precision .* (val )))
	model = G* ChainRulesCore.ignore_derivatives(amplitude) 
	return likelihood(data, model)
end


"""
    find_peaks!(vect) -> inds

yields the indices of the most significant local maxima found in vector `vect`
destroying the contents of `vect` in the process.  The indices are returned in
decreasing order of peak heights.

The following keywords can be used to tune the algorithm:

* `dist` (5 by default) specifies the minimal distance (in indices units)
  between two peaks.

* `atol` (0.0 by default) and `rtol` (0.1 by default) specify the absolute and
  relative tolerances for the detection threshold.  If `atol` is NaN, the
  detection threshold is `rtol*maximum(vect)`; otherwise, the detection
  threshold is `max(atol, rtol*maximum(vect))`.  All selected peaks have values
  greater or equalt the detection threshold.

* `nmax` (no limits by default) specifies the maximum number of peaks to
  detect.

Call `find_peaks` to avoid overwriting `vect`.

"""
function find_peaks!(vect::AbstractVector{<:Real};
                     dist::Integer = 5,
                     rtol::Real = 0.1,
                     atol::Real = 0.0,
                     nmax::Integer = typemax(Int))
    dist ≥ 1 || error("minimal distance must be ≥ 1")
    0 ≤ rtol ≤ 1 || error("invalid relative threshold")

    dst = Int(dist) - 1
    vmin = typemin(eltype(vect))
    vtol = float(atol)
    I = axes(vect, 1)
    I_first, I_last = first(I), last(I)
    inds = Int[]
    while length(inds) < nmax
        vmax, imax = findmax(vect)
        if length(inds) < 1
            # Compute selection threshold.
            v = oftype(vtol, rtol*vmax)
            if isnan(vtol)
                vtol = v
            else
                vtol = max(vtol, v)
            end
        end
        vmax ≥ vtol || break
        push!(inds, imax)
        @inbounds for i in max(I_first, imax - dst):min(I_last, imax + dst)
            vect[i] = vmin
        end
    end
    return inds
end

"""
    find_peaks(vect) -> inds

yields the indices of the most significant local maxima found in vector `vect`.

See `find_peaks!` for a list of accepted keywords.

"""
find_peaks(vect::AbstractVector, args...; kwds...) =
    find_peaks!(copy(vect), args...; kwds...)

#= function gravi_spectral_calibration(wave::AbstractWeightedData{T, 2},darkwave::AbstractWeightedData{T, 2}, profile::Dict{String,<:Profile}; lines=argon[:,1],hw=2,λorder=3) where T

       wav = gravi_extract_profile(wave - darkwave, profile)
#=        argonspectrum =  sum(values(wav)) / length(wav)
       argonpeaks = sort!(find_peaks(argonspectrum.val,rtol=0.0,nmax=12))

       continuum = deepcopy(argonspectrum)
       flagbadpix!(continuum, [any( max(n-3,1) .<= argonpeaks .< min(length(continuum),n+3)) for n ∈ 1:(length(continuum))])

       continuum = trues(size(argonspectrum))
       continuum[[any( max(n-3,1) .<= argonpeaks .< min(length(continuum),n+3)) for n ∈ 1:(length(continuum))]] .= false;

       knt = SVector{18,Float32}(1.0, 24.0, 35.0, 41.0, 46.0, 58.0, 69.0, 91.0, 114.0, 125.0, 136.0, 159.0, 181.0, 226.0, 271.0, 294.0, 316.0, 360.0)
	#sp4 = Spline1D(1:360, meanspectrum.val; w=meanspectrum.precision, k=3, bc="zero",s=0.01)
	B = BSplineBasis(BSplineOrder(3), knt)
	ncoefs = length(B)
       nspectra = length(argonpeaks)
	coefs = [ [zeros(T,3)...,ones(T,ncoefs-6)...,zeros(T,3)...] for i ∈ 1:nspectra]
       
       chnl= "12-A-C"
       for p ∈ argonpeaks
              σ = p .^(0:(5-1))'* profile[chnl].σ
              rng =  max(p-3,1):min(length(argonspectrum),p+3)
              param =(;center= [Float64(p)], σ=σ)
              flatparam, restructure = destructure(param)
              func = build_func(view(wav[chnl],rng),rng,restructure) 
              res = optimize(func, flatparam, NelderMead(),Optim.Options(iterations=100))
              restructure(Optim.minimizer(res))
       end
 =#    
       Threads.@threads for tel1 ∈ 1:4
              for tel2 ∈ 1:4
                     tel1==tel2 && continue
                     for chnl ∈ ["A","B","C","D"]
                            chname = "$tel1$tel2-$chnl-C"
                            haskey(profile,chname) || continue
                            updatedprofile = gravi_spectral_calibration(wav[chname] ,profile[chname];hw=hw, lines=lines,λorder=λorder)
                            @show updatedprofile
                            push!(profile,chname=>updatedprofile) 
                     end
              end
       end
       return profile
end =#

function gravi_spectral_calibration(wave::AbstractWeightedData{T,1}, profile::Profile{A}; lines=argon[:,1],hw=2,λorder=3)  where {A,T}
       #argonpeaks = sort!(find_peaks(wave.val,rtol=0.0,nmax=12))
       argonpeaks = argon[:,2]
       np = length(argonpeaks)
       position = Vector{Float64}(undef,np)
       width = Vector{Float64}(undef,np)
       σdeg = length(profile.σ)
       for (i,p) ∈ enumerate(argonpeaks)
              σ = p .^(0:(σdeg-1))'* profile.σ
              rng =  max(round(Int,p-hw),1):min(length(wave),round(Int,p+hw))
              @show (;center, σ) =  fit_peak(rng,view(wave,rng); center=[Float64(p)],σ=σ)
              #@show J = error_estimation(rng,view(wave,rng);center= center, σ=σ)
              position[i] = center[1]
              width[i] = σ[1]
       end
       #poly = align(lines,position)
       #predicted_pixels = poly[1] .+ poly[2] .* lines
       predicted_pixels = argon[:,2]
       doublets = diff(predicted_pixels) .<(hw)

       fitted_pixels = collect(copy(predicted_pixels))

       iter = enumerate(predicted_pixels)
       next = iterate(iter)
       while next !== nothing
              ((i,pos), state) = next
              if i<length(lines) && doublets[i] 
                     rng=max(1,round(Int,pos)-hw) :min(round(Int,predicted_pixels[i+1])+hw,length(wave))
                     σ = (pos .^(0:(σdeg-1))'* profile.σ)
                     center = [predicted_pixels[i],predicted_pixels[i+1]]
                     (;center, σ) =  fit_peak(rng,view(wave,rng); center=center,σ=σ)
#                     @show J = error_estimation(rng,view(wave,rng);center= center, σ=σ)
                     fitted_pixels[i]  = center[1]
                     fitted_pixels[i+1]  = center[2]
                     position[map(x->x∈rng,round.(Int,position))] .= 0
                     (_, state)  = iterate(iter, state) #skip both lines
                     next = iterate(iter, state)

                     continue
              end 

              tmp = pos -hw .< position .< pos +hw
              if sum(tmp) == 1 # correct line identification
                     fitted_pixels[i] = position[findfirst(tmp)]
                     #position[tmp] .= 0
              elseif sum(tmp) == 0 # undetected line
                     rng= max(1,round(Int,pos)-hw) : min(round(Int,pos)+hw,length(wave))
                     σ = (pos .^(0:(σdeg-1))'* profile.σ) 
                     center = [pos]
                     (;center, σ) =  fit_peak(rng,view(wave,rng); center=center,σ=σ)
                     #@show J = error_estimation(rng,view(wave,rng);center= center, σ=σ)

                     fitted_pixels[i]  = center[1]
              else
                     fitted_pixels[i] = position[findfirst(tmp)+argmin(position[tmp] .- pos)-1]
                     #position[tmp] .= 0
              end
              next = iterate(iter, state)
       end

       #remove first line (usually badly fitted)
       P = hcat( (fitted_pixels[2:end].^n for n=0:3)...)

       @show @reset profile.λ = inv(P'*P)*P'*lines[2:end]
       return profile
       #return fitted_pixels
end

function error_estimation(rng,data;center= 0.0, σ=0.5)
       param =(;center= center)
       flatparam, restructure = ParameterHandling.flatten(param)
     #  val = gaussian_lines(rng;restructure(flatparam)...)
       val,grad = Zygote.withjacobian(x->gaussian_lines(rng;restructure(x)...,σ=σ), flatparam)
       val=reshape(val,length(data),:)
       G =hcat(val,ones(length(data)))
       amplitude = getamplitude(data,G)
       model = G* amplitude 
       α = length(rng)/likelihood(data,model)

       J =first(grad).*amplitude[1:end-1]'

       inv(J'*(α.*data.precision .* J))
end


function fit_peak(rng,data;center= 0.0, σ=0.5 ,centeronly=true)
       if centeronly
              param =(;center= center)
              flatparam, restructure = ParameterHandling.flatten(param)
              func = build_func(data,rng,restructure; σ=σ)               
       else
              param =(;center= center, σ=σ)
              flatparam, restructure = ParameterHandling.flatten(param)
              func = build_func(data,rng,restructure) 
       end
     #  res = optimize(func, flatparam, BFGS(),Optim.Options(iterations=1000))
       x = vmlmb(func, flatparam; ftol = (0.0,1e-12),lower= minimum(rng), upper = maximum(rng),maxeval=500,autodiff=true);
       if centeronly
            #  (;center)  =  restructure(Optim.minimizer(res))
              (;center)  =  restructure(x)

       else
              (;center, σ)  =  restructure(x)
       end
       return (;center, σ)
end

function build_func(data,rng,restructure;σ=nothing )
       isnothing(σ) &&  return x-> loss(data,rng;restructure(x)...)
       return x-> loss(data,rng;σ=σ, restructure(x)...)
end


function loss(data::AbstractWeightedData{T,1}, rng;center=[Float64(0)],σ=[Float64(1)] ) where T
       #(;val, precision)=data
       n =length(data)
       G = hcat(gaussian_lines(rng;center=center,σ=σ), ones(n))#, 2 .*(1:n)./(n*(n+1)))
       #amplitude = pinv(G' * ( data.precision.* G))*G'* (data.precision .* (data.val ))
       amplitude = getamplitude(data,G)
       model = G* amplitude 
	return likelihood(data,model)
end

 function getamplitude(data::AbstractWeightedData,model)
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


function align(lines::AbstractVector,pixels::AbstractVector; rtol=0.01) 
       la = length(lines)
       lb = length(pixels)

       M = zeros(Int,la,lb)
       @inbounds for ia ∈ 1:la-2
              for ja ∈ ia+1:la-1
                     for ka ∈ ja+1:la
                            for ib ∈ 1:lb-2
                                   for jb ∈ ib+1:lb-1
                                          for kb ∈ jb+1:lb
                                                 test = isapprox((lines[ja] - lines[ia])/(lines[ka] - lines[ja]), (pixels[jb] - pixels[ib])/(pixels[kb] - pixels[jb]);rtol=rtol) 
                                                 M[ia,ib] += test
                                                 M[ja,jb] += test
                                                 M[ka,kb] += test
                                          end
                                   end
                            end
                     end
              end
       end
       max1 = argmax(M)
       M[max1] =0
       max2 = argmax(M)
       M[max2] =0
       max3 = argmax(M)
       M[max3] =0
       max4 = argmax(M)
       M[max4] =0
       max5 = argmax(M)

       pixels = pixels[[max1[2],max2[2],max3[2],max4[2],max5[2]]]
       lines = lines[[max1[1],max2[1],max3[1],max4[1],max5[1]]]

       P = [lines.^0 lines.^1]

       return inv(P'*P)*P'*pixels
       
end 



#= 
function align(a::AbstractVector,b::AbstractVector; rtol=0.01) 
       la = length(a)
       lb = length(b)
       goodA  = falses(la)
       goodB  = falses(la)

       da = collect(diff(a))
       db = collect(diff(b))

       M = zeros(Int,la-1,lb-1)
       @inbounds for pa = 1:(la-1)
              @inbounds for pb = 1:(lb-1)
                     tmpM = hcat(map(x->isapprox.(x, da./da[pa]; rtol=rtol),db./db[pb])...)
                     tmpM[pa,pb] = false
                     M .+= tmpM
              end
       end
       return M
end

function dynamicprogram(M)
       cost = Float64.(M)
       pred = Array{CartesianIndex{2},2}(undef,size(M))
       indexM = CartesianIndices(M)
       Neighborhood= [      CartesianIndex(-3, -3)  CartesianIndex(-3, -2)  CartesianIndex(-3, -1)
                            CartesianIndex(-2, -3)  CartesianIndex(-2, -2)  CartesianIndex(-2, -1)
                            CartesianIndex(-1, -3)  CartesianIndex(-1, -2)  CartesianIndex(-1, -1)]
       for i ∈ indexM
              costi = cost[i]
              pred[i] = CartesianIndex(0, 0) 
              for j ∈ Neighborhood
                     k = i + j
                     #d =  0.1*(1 .- sqrt(sum(j.I.^2))/5)
                     if k ∈ indexM
                            if (cost[k]+costi) > cost[i]
                                   cost[i] = (cost[k]+costi)
                                   pred[i] = k
                            end
                     end
              end  
       end
       return (cost, pred)
end
 =#