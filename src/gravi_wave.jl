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

function gravi_spectral_calibration(      wave::AbstractWeightedData{T,1}, 
                                          profile::SpectrumModel{A,B,Nothing}; 
                                          lines=argon[:,1], 
                                          guess=argon[:,2],
                                          λorder=3)  where {A,B,T}


       P = hcat( ((lines .* 1e6).^n for n=0:λorder)...)
       init = inv(P'*P)*P' * guess
       s = profile.σ
       f(x) = loss(wave,s,P, x)
       x = vmlmb(f, init;maxeval=500,ftol=(0,0), autodiff=true);
       Q = hcat( ((P*x).^n for n=0:3)...)
       @reset profile.λ = collect(inv(Q'*Q)*Q'*lines )
end

function loss(data::AbstractWeightedData{T,1}, prσ::AbstractVector,P::AbstractMatrix,x::AbstractVector) where T
       σdeg = length(prσ)
       rng = axes(data,1)
       prediction = P * x
       σ = prediction .^(0:(σdeg-1))'* prσ
       G = gaussian_lines(rng;center=prediction,σ=σ)
       amp = max.(0.,getamplitude(data,G))
       likelihood(data,G*amp)

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