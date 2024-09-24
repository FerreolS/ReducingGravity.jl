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
end
P2VM = Dict(P2VM)

flats,darkp2vm,p2vm,gp,chnames = ReducingGravity.gravi_reorder_p2vm(P2VM,bboxes,illuminated,goodpix; filterblink=true,keepbias=true,blinkkernel=9)
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

spctr = gravi_extract_profile_flats_from_p2vm(flats.-[darkp2vm],chnames,profiles)

profiles, lamp = gravi_compute_lamp_transmissions(  spctr, profiles; nb_transmission_knts=300,nb_lamp_knts=300, Chi2=0.5,restart=true)

ron = gravi_compute_ron(darkp2vm,goodpix,gain)
wd = gravi_create_weighteddata(p2vm, illuminated,goodpix, gain,ron)
tλ = ReducingGravity.build_wavelength_range(profiles)

#tλ4 = tλ[1:4:end]
#itrp4 = ReducingGravity.Interpolator(tλ4,CatmullRomSpline{Float32}())

itrp = ReducingGravity.Interpolator(tλ,CatmullRomSpline{Float32}())
baseline_phasors, baseline_visibilities =  gravi_build_p2vm_interf(wd - darkp2vm,itrp, profiles,lamp;loop=5, rgl_phasor=1000,rgl_vis=1000)

S,tλ,wvidx = gravi_build_V2PM(profiles,baseline_phasors;λmin=2e-6,λmax=2.5e-6)

wvscfits = read(FITS(first(filter(x -> (occursin(r"(WAVE,SC)", x.second.type)), flist)).first)["IMAGING_DATA_SC"]);
wvsc =  gravi_create_weighteddata(wvscfits, illuminated,goodpix,gain,ron)
wvsc = ReducingGravity.make_pixels_vector(wvsc - darkflat,profiles,wvidx);
#wvsc1 = view(wvsc,:,1:5)
wvcorr = ReducingGravity.get_correlatedflux(S,wvsc)
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

ron3 = gravi_compute_ron(dark3,goodpix, gain)
object3 = read(FITS(first(filter(x -> (occursin(r"(OBJECT)", x.second.type) && x.second.Δt==3.0), flist)).first)["IMAGING_DATA_SC"]);
object3 = gravi_create_weighteddata(object3,illuminated, goodpix3.&&goodpix,  gain, ron3)
