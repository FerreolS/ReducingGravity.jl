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
                                          profile::SpectrumModel; 
                                          lines=argon[:,1], 
                                          guess=argon[:,2],
                                          λorder=3)  where T


       P = hcat( ((lines .* 1e6).^n for n=0:λorder)...)
       init = inv(P'*P)*P' * guess
       s = profile.σ
       f(x) = loss(wave,mean(s,dims=2),P, x)
       x = vmlmb(f, init;maxeval=500,ftol=(0,0), autodiff=true);
       Q = hcat( ((P*x).^n for n=0:3)...)
       λcoefs = collect(inv(Q'*Q)*Q'*lines )
       return   add_spectral_law(profile,λcoefs)
end

function loss(data::AbstractWeightedData{T,1}, prσ::AbstractArray,P::AbstractMatrix,x::AbstractVector) where T
       σdeg = size(prσ,1)
       rng = axes(data,1)
       prediction = P * x

       σ = prediction .^(0:(σdeg-1))'* prσ
       
       G = gaussian_lines(rng;center=prediction,σ=σ)
       amp = max.(0.,getamplitude(data,G))
       likelihood(data,G*amp)

end


function add_spectral_law(s::SpectrumModel,λcoefs) 
	p = s.bbox.indices[1]
	λdeg = length(λcoefs)
 	λ = p .^(0:(λdeg-1))'* λcoefs
	cdeg = length(s.center)
	P = (λ).^(0:(cdeg-1))'
	cntr = get_center(s)
	new_center = inv(P'*P)*P'* cntr

	σdeg = size(s.σ,1)
	P = (λ).^(0:(σdeg-1))'
	sgm = get_width(s)
	new_σ = inv(P'*P)*P'* sgm
	return SpectrumModel(new_center,new_σ,λcoefs,[0.,+Inf],Vector{InterpolatedSpectrum{Nothing,Nothing}}(),s.bbox)
end



function gravi_spectral_calibration_pipeline(wavefits::FitsFile,
						profiles::Dict{String,SpectrumModel{A,B, C, E}}; 
                                          ) where {A,B,C,E}
       endswith(wavefits[1]["PIPEFILE"].string,"_wave.fits") || error("must be _wave.fits file")

       WAVE_DATA_SC = wavefits["WAVE_DATA_SC"]
       STARTX = WAVE_DATA_SC["ESO PRO PROFILE STARTX"].value(Int)
       NX = WAVE_DATA_SC["ESO PRO PROFILE NX"].value(Int)
       wavedata = read(WAVE_DATA_SC)

       IMAGING_DETECTOR_SC = read(wavefits["IMAGING_DETECTOR_SC"])
       regname = IMAGING_DETECTOR_SC["REGNAME"]
       region = IMAGING_DETECTOR_SC["REGION"]

       new_profiles = Dict{String,SpectrumModel{A,Vector{Float64}, C,E}}()

       nλ = size(first(values(profiles)).bbox,1)
       for (reg,name) ∈ zip(region,regname)
              wv = Vector{Float64}(undef,nλ)
              fill!(wv,NaN)
              wvd = wavedata["DATA$reg"][:]

              λmin = minimum(wvd)
		λmax = maximum(wvd)
              wv[STARTX:(STARTX+NX-1)] .= wvd
              profile = profiles[name]
              @reset profile.λbnd = [λmin, λmax]
              @reset profile.λ = wv
              push!(new_profiles,name=>profile)
       end


	return new_profiles
end
