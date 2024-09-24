using FITSIO,Statistics, ArrayTools,StatsBase, LinearAlgebra,InterpolationKernels
using ReducingGravity


dirpath = "/Users/ferreol/Data/Gravity+/2020-01-06_MEDIUM_COMBINED/"

#dirpath = "/Users/ferreol/Data/RawData/GRAVITY+/2018-03-09_HIGH_COMBINED"
#dirpath = "/Users/ferreol/Data/Gravity+/2020-03-09_MEDIUM_COMBINED/"
flist = ReducingGravity.listfitsfiles(dirpath);

# Illumination size estimation
Δtflat = first(filter(x -> occursin("FLAT1", x.second.type), flist)).second.Δt
fdark = FITS(first(filter(x -> (occursin(r"(DARK)", x.second.type) && x.second.Δt==Δtflat), flist)).first);
(illuminated,bboxes) = gravi_data_create_bias_mask(fdark);

# Bad pixel detection
darkflat = read(fdark["IMAGING_DATA_SC"]);
goodpix = gravi_compute_badpix(darkflat,illuminated,spatialkernel=(11,1))
illuminated = illuminated .|| .!goodpix
darkflat,goodpix  = gravi_create_weighteddata(darkflat,illuminated,goodpix)

# Flat dark and P2VM construction from P2VM files
len_p2vm = length(filter(x -> occursin("P2VM", x.second.type), flist))
P2VM = Vector{Pair{String, Array{Float32, 3}}}(undef,len_p2vm)
#P2VMwd = Vector{Pair{String,ConcreteWeightedData{Float32,2}}}(undef,len_p2vm)

Threads.@threads for (i,pv2mfile) ∈ collect(enumerate(values(filter(x -> occursin("P2VM", x.second.type), flist))))
	rawdata =read(FITS(pv2mfile.path)["IMAGING_DATA_SC"])
	P2VM[i] = pv2mfile.type => rawdata#gravi_data_detector_cleanup(rawdata,illuminated,keepbias=true)[2]
#	data,_ = gravi_create_weighteddata(rawdata,illuminated,goodpix; filterblink=true, blinkkernel=9,keepbias=true, cleanup=true)
#	P2VMwd[i] = pv2mfile.type =>data
end
P2VM = Dict(P2VM)
#P2VMwd = Dict(P2VMwd)

flats,darkp2vm,p2vm,gp,chnames = ReducingGravity.gravi_reorder_p2vm(P2VM,bboxes,illuminated,goodpix; filterblink=true,keepbias=true,blinkkernel=9)
#flats,darkp2vm,p2vm,gp = ReducingGravity.gravi_compute_flat_and_dark_from_p2vm(P2VM,bboxes,illuminated,goodpix; filterblink=true,keepbias=true)
profiles = gravi_compute_profile(combine(flats .- [darkp2vm]),bboxes,thrsld=0.1);


# spectral calibration
Δtwave = first(filter(x -> occursin(r"(WAVE,LAMP)", x.second.type), flist)).second.Δt
darkwave = read(FITS(first(filter(x -> (occursin(r"(DARK)", x.second.type) && x.second.Δt==Δtwave ), flist)).first)["IMAGING_DATA_SC"]);
darkwave,goodpix  = gravi_create_weighteddata(darkwave,illuminated,goodpix)

fwave = FITS(first(filter(x -> (occursin(r"(WAVE,LAMP)", x.second.type) ), flist)).first);
wave =read(fwave["IMAGING_DATA_SC"]);
wave,goodpix  = gravi_create_weighteddata(wave,illuminated,goodpix)

profiles = gravi_spectral_calibration(wave,darkwave, profiles; nonnegative=true, robust=false);





gain, rov = gravi_compute_gain_from_p2vm(flats,darkp2vm,profiles,goodpix)
#spctr = gravi_extract_profile_flats_from_p2vm(P2VMwd , darkp2vm, profiles; nonnegative=true, robust=true)
# TODO To be adapted once the sorted gravi_compute_gain_from_p2vm works
#darkp2vm, gain, rov = gravi_compute_gain_from_p2vm(P2VMwd,profiles,goodpix)
#spctr = gravi_extract_profile_flats_from_p2vm(P2VMwd , darkp2vm,profiles; nonnegative=true, robust=false)
spctr = gravi_extract_profile_flats_from_p2vm(flats.-[darkp2vm],chnames,profiles)
#profiles, lamp = gravi_compute_lamp_transmissions(  spctr, profiles; nb_transmission_knts=50,nb_lamp_knts=300, Chi2=1.)
profiles, lamp = gravi_compute_lamp_transmissions(  spctr, profiles; nb_transmission_knts=300,nb_lamp_knts=300, Chi2=0.5,restart=true)

ron = gravi_compute_ron(darkp2vm,goodpix,gain)
wd = gravi_create_weighteddata(p2vm, illuminated,goodpix, gain,ron)
tλ = ReducingGravity.build_wavelength_range(profiles)

#tλ4 = tλ[1:4:end]
#itrp4 = ReducingGravity.Interpolator(tλ4,CatmullRomSpline{Float32}())

itrp = ReducingGravity.Interpolator(tλ,CatmullRomSpline{Float32}())
baseline_phasors, baseline_visibilities =  gravi_build_p2vm_interf(wd - darkp2vm,itrp, profiles,lamp;loop=5, rgl_phasor=1000,rgl_vis=1000)
#gravi_build_p2vm_interf(wd - darkp2vm,profiles,lamp; loop_with_norm=10, loop=2)
S,tλ,wvidx = gravi_build_V2PM(profiles,baseline_phasors;λmin=2e-6,λmax=2.5e-6)
#wdp = ReducingGravity.make_pixels_vector(view(p12,:,:,100) - darkflat,profiles,wvidx);

#fcorr = ReducingGravity.get_correlatedflux(S,wdp)


#=A  = ReducingGravity.gravi_extract_channel(wd-darkp2vm,profiles["13-A-C"],lamp)
B = ReducingGravity.gravi_extract_channel(wd-darkp2vm,profiles["13-B-C"],lamp)
C = ReducingGravity.gravi_extract_channel(wd-darkp2vm,profiles["13-C-C"],lamp)
D = ReducingGravity.gravi_extract_channel(wd-darkp2vm,profiles["13-D-C"],lamp)
ϕ = ReducingGravity.gravi_initial_input_phase(A,B,C,D)
phasors= ReducingGravity.gravi_build_ABCD_phasors(ϕ,A,B,C,D);

phase = ReducingGravity.estimate_visibility(phasors,A,B,C,D);
v=phase;
rho = sqrt.(phase[1,:,:].^2 .+ phase[2,:,:] .^2)
rho3 = (ones(360) .* median(rho[50:200,:],dims=1))
phase .*= reshape(1 ./ rho  .* rho3,1,size(rho)...)
phasors= ReducingGravity.gravi_build_ABCD_phasors(phase,A,B,C,D);
#phasors[phasors.>1.].=0.
#phasors[phasors.<0.8].=0.
phase = ReducingGravity.estimate_visibility(phasors,A,B,C,D);
sum(abs2,filter(isfinite,(phase.-v))) 
 =#

 if false

#S2,tλ,λbaseline,wvidx = ReducingGravity.gravi_build_p2vm_matrix(profiles,baseline_phasors; λmin=2e-6,λmax=2.5e-6);
p12 = gravi_create_weighteddata(P2VM["P2VM12"], illuminated,goodpix,rov, gain)
wdp = ReducingGravity.make_pixels_vector(view(p12,:,:,100) - darkflat,profiles,wvidx);
Cx = pinv(Symmetric(Array(S'*(wdp.precision.*S))))
xx = Cx*S'*(wdp.precision.* wdp.val);
xx = reshape(xx,6*2+4,:);
ww = sqrt(Symmetric(Cx))
stdxx = diag(ww)

using KrylovKit
A = Symmetric((S'*(wdp.precision.*S)))
b = S'*(wdp.precision.*wdp.val);
xx,info= KrylovKit.linsolve(A,b[:]; issymmetric=true, maxiter=100,atol=1e-3);

S,tλ,wvidx = gravi_build_V2PM(profiles,baseline_phasors)
#S,tλ,wvidx = gravi_build_V2PM(profiles,baseline_phasors;λmin=2e-6,λmax=2.5e-6)
wvscfits = read(FITS(first(filter(x -> (occursin(r"(WAVE,SC)", x.second.type)), flist)).first)["IMAGING_DATA_SC"]);
wvsc = gravi_create_weighteddata(wvscfits, illuminated,goodpix,rov, gain)
wvsc = ReducingGravity.make_pixels_vector(wvsc - darkflat,profiles,wvidx);
wvsc1 = view(wvsc,:,1:5)
wvcorr = ReducingGravity.get_correlatedflux(S,wvsc1)
photometric,interferometric = ReducingGravity.extract_correlated_flux(wvcorr)
bispectra = ReducingGravity.get_bispectrum(interferometric)
clcorr=ReducingGravity.get_closure_correction(bispectra)
S,λ,wvidx = gravi_build_V2PM(profiles,baseline_phasors;λsampling=λ,λmin=1.96e-6,λmax=2.5e-6, closure_correction=clcorr)

fdark3 = read(FITS(first(filter(x -> (occursin(r"(DARK)", x.second.type) && x.second.Δt==3.0), flist)).first)["IMAGING_DATA_SC"]);
goodpix3 = gravi_compute_badpix(fdark3,illuminated, spatialkernel=(11,1))
dark3,goodpix = gravi_create_weighteddata(fdark3,illuminated,goodpix3.&&goodpix; filterblink=true, blinkkernel=(9),keepbias=true)

sky3 = read(FITS(first(filter(x -> (occursin(r"(SKY)", x.second.type) && x.second.Δt==3.0), flist)).first)["IMAGING_DATA_SC"]);
goodpix3 = gravi_compute_badpix(sky3,illuminated, spatialkernel=(11,1))
sky3,goodpix = gravi_create_weighteddata(sky3,illuminated,goodpix3.&&goodpix; filterblink=true, blinkkernel=(5),keepbias=true)

object3 = read(FITS(first(filter(x -> (occursin(r"(OBJECT)", x.second.type) && x.second.Δt==3.0), flist)).first)["IMAGING_DATA_SC"]);
object3 = gravi_create_weighteddata(object3,illuminated, goodpix3.&&goodpix, rov, gain)

 end