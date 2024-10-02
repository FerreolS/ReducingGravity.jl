module ReducingGravityPlotsExt
using Plots

@recipe function f((;val,precision)::AbstractWeightedData)
	extval = extrema(val)
	σ = 3 .*sqrt.(1 ./ precision)
	σ[ σ.== Inf] .= extval[2] - extval[1]
	ribbon := σ
	fillalpha := 0.5
	ylims := extval
	val
end


@recipe function f(x,(;val,precision)::AbstractWeightedData)
	extval = extrema(val)
	σ = 3 .*sqrt.(1 ./ precision)
	σ[ σ.== Inf] .= extval[2] - extval[1]
	ribbon := σ
	fillalpha := 0.5
	ylims := extval
	(x,val)
end
end