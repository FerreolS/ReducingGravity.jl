module ReducingGravityPlotsExt
using Plots

@recipe function f((;val,precision)::AbstractWeightedData)
	maxval = maximum(val)
	σ = 3 .*sqrt.(1 ./ precision)
	σ[ σ.== Inf] .= maxval
	ribbon := σ
	fillalpha := 0.5
	ylims := extrema(val)
	val
end


@recipe function f(x,(;val,precision)::AbstractWeightedData)
	maxval = maximum(val)
	σ = 3 .*sqrt.(1 ./ precision)
	σ[ σ.== Inf] .= maxval
	ribbon := σ
	fillalpha := 0.5
	ylims := extrema(val)
	(x,val)
end
end