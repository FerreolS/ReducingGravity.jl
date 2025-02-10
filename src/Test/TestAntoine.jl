using Revise, EasyFITS,Statistics, ArrayTools,StatsBase, LinearAlgebra,InterpolationKernels
using ReducingGravity


dirpath = "/Users/ferreol/Data/RawData/GRAVITY+/AntoineMerand"
wavepath = "/Users/ferreol/Data/RawData/GRAVITY+/AntoineMerand/reduced_vfactor/GRAVI.2021-10-01T12:04:42.600_wave.fits"

flist = ReducingGravity.listfitsfiles(dirpath);

# Illumination size estimation
Δtflat = first(filter(x -> occursin("FLAT1", x.second.type), flist)).second.Δt
fdark = openfits(first(filter(x -> (occursin(r"(DARK)", x.second.type) && x.second.Δt==Δtflat), flist)).first);
(illuminated,bboxes) = gravi_data_create_bias_mask(fdark);

# Bad pixel detection
darkflat = read(fdark["IMAGING_DATA_SC"]);
goodpix = gravi_compute_badpix(darkflat,illuminated,spatialkernel=(11,1))
illuminated = illuminated .|| .!goodpix
darkflat,	  = gravi_create_weighteddata(darkflat,illuminated,goodpix)

# Flat dark and P2VM construction from P2VM files
len_p2vm = length(filter(x -> occursin("P2VM", x.second.type), flist))
P2VM = Vector{Pair{String, Array{Float32, 3}}}(undef,len_p2vm)
#P2VMwd = Vector{Pair{String,ConcreteWeightedData{Float32,2}}}(undef,len_p2vm)

Threads.@threads for (i,pv2mfile) ∈ collect(enumerate(values(filter(x -> occursin("P2VM", x.second.type), flist))))
	rawdata =readfits(pv2mfile.path; ext="IMAGING_DATA_SC")
	P2VM[i] = pv2mfile.type => rawdata#gravi_data_detector_cleanup(rawdata,illuminated,keepbias=true)[2]
end
P2VM = Dict(P2VM)

flats,darkp2vm,p2vm,goodpix,chnames = ReducingGravity.gravi_reorder_p2vm(P2VM,bboxes,illuminated,goodpix; filterblink=true,keepbias=true,blinkkernel=9)
profiles = gravi_compute_profile(combine(flats .- [darkp2vm]),bboxes,thrsld=0.1);

# spectral calibration

# Δtwave = first(filter(x -> occursin(r"(WAVE,LAMP)", x.second.type), flistdark)).second.Δt
# darkwave = readfits((first(filter(x -> (occursin(r"(DARK)", x.second.type) && x.second.Δt==Δtwave ), flistdark)).first; ext="IMAGING_DATA_SC");
# darkwave,goodpix  = gravi_create_weighteddata(darkwave,illuminated,goodpix)

# fwave = FITS(first(filter(x -> (occursin(r"(WAVE,LAMP)", x.second.type) ), flistdark)).first);
# wave =read(fwave["IMAGING_DATA_SC"]);
# wave,goodpix  = gravi_create_weighteddata(wave,illuminated,goodpix)

# profiles = gravi_spectral_calibration(wave,darkwave, profiles; nonnegative=true, robust=false);

profiles = ReducingGravity.gravi_spectral_calibration_pipeline(openfits(wavepath),profiles)




gain, rov = gravi_compute_gain_from_p2vm(flats,darkp2vm,profiles,goodpix)

spctr = gravi_extract_profile_flats_from_p2vm(flats.-[darkp2vm],chnames,profiles)

profiles, lamp = gravi_compute_lamp_transmissions(  spctr, profiles; nb_transmission_knts=300,nb_lamp_knts=300, Chi2=0.5,restart=true)

ron = gravi_compute_ron(darkp2vm,goodpix,gain)
wd = gravi_create_weighteddata(p2vm, illuminated,goodpix, gain,ron)
tλ = ReducingGravity.build_wavelength_range(profiles)

tλ4 = tλ[1:4:end]
itrp4 = ReducingGravity.Interpolator(tλ4,CatmullRomSpline{Float32}())

itrp = ReducingGravity.Interpolator(tλ,CatmullRomSpline{Float32}())

p2v = gravi_extract_profile(wd - darkp2vm.val,profiles )
baseline_phasors, baseline_visibilities = ReducingGravity.gravi_build_p2vm_interf_flat(p2v,spctr, itrp4, profiles, lamp,loop=15)
#baseline_phasors, baseline_visibilities =  gravi_build_p2vm_interf(wd - darkp2vm,itrp, profiles,lamp;loop=5, rgl_phasor=100,rgl_vis=100)

S,λ,wvidx = gravi_build_V2PM(profiles,baseline_phasors)#;λmin=2e-6,λmax=2.5e-6)


wvscfits = readfits(first(filter(x -> (occursin(r"(WAVE,SC)", x.second.type)), flist)).first; ext="IMAGING_DATA_SC");
wvsc =  gravi_create_weighteddata(wvscfits, illuminated,goodpix,gain,ron)
wvsc = ReducingGravity.make_pixels_vector(wvsc - darkflat,profiles,wvidx);
#wvsc1 = view(wvsc,:,1:5)
wvcorr = ReducingGravity.get_correlatedflux(S,wvsc)
wvphotometric,wvinterferometric = ReducingGravity.extract_correlated_flux(wvcorr)
bispectra = ReducingGravity.get_bispectrum(wvinterferometric)
clcorr=ReducingGravity.get_closure_correction(bispectra)
closure_correction = ReducingGravity.make_closure_correction(clcorr)
S = S*closure_correction 
wvcorr = ReducingGravity.get_correlatedflux(S,wvsc)
wvphotometric,wvinterferometric = ReducingGravity.extract_correlated_flux(wvcorr)

disp_filename = "/Users/ferreol/Code/gravity_dev/gravity-calib/GRAVI_DISP_MODEL_2019-10-17.fits"
dispmodel=ReducingGravity.gravi_extract_disp_model(disp_filename)
dnorm = ReducingGravity.normalize_data(wvsc,S,wvphotometric)
fλ = ReducingGravity.recompute_wavelegnth(wvinterferometric,λ;lmin=15,lmax=200)
nkt, phasorst,opl = ReducingGravity.recalibrate(dnorm, wvinterferometric , dispmodel, fλ, profiles; iter=4)
#S,λ,wvidx = gravi_build_V2PM(profiles,baseline_phasors;λsampling=λ,closure_correction=clcorr)

fdark30 = readfits(first(filter(x -> (occursin(r"(DARK)", x.second.type) && x.second.Δt==30.0), flist)).first; ext="IMAGING_DATA_SC");
goodpix30 = gravi_compute_badpix(fdark30,illuminated, spatialkernel=(11,1))
dark30,goodpix = gravi_create_weighteddata(fdark30,illuminated,goodpix30.&&goodpix; filterblink=true, blinkkernel=(9),keepbias=true)

sky30o = readfits("/Users/ferreol/Data/RawData/GRAVITY+/AntoineMerand/GRAVI.2021-10-02T06:32:00.698.fits"; ext="IMAGING_DATA_SC");
goodpix30 = gravi_compute_badpix(sky30o,illuminated, spatialkernel=(11,1))
sky30o,goodpix = gravi_create_weighteddata(sky30o,illuminated,goodpix30.&&goodpix; filterblink=true, blinkkernel=(5),keepbias=true)

ron30 = gravi_compute_ron(dark30,goodpix, gain)
object30 = readfits(first(filter(x -> (occursin(r"(OBJECT)", x.second.type) && x.second.Δt==30.0), flist)).first; ext="IMAGING_DATA_SC");
object30 = gravi_create_weighteddata(object30,illuminated, goodpix30.&&goodpix,  gain, ron30)


sky30c = readfits("/Users/ferreol/Data/RawData/GRAVITY+/AntoineMerand/GRAVI.2021-10-02T06:45:48.733.fits"; ext="IMAGING_DATA_SC");
goodpix30 = gravi_compute_badpix(sky30c,illuminated, spatialkernel=(11,1))
sky30c,goodpix = gravi_create_weighteddata(sky30c,illuminated,goodpix30.&&goodpix; filterblink=true, blinkkernel=(5),keepbias=true)

calib = readfits(first(filter(x -> (occursin(r"(STD)", x.second.type) && x.second.Δt==30.0), flist)).first; ext="IMAGING_DATA_SC");
calib = gravi_create_weighteddata(calib,illuminated, goodpix30.&&goodpix,  gain, ron30)

objsc = ReducingGravity.make_pixels_vector(object30 - sky30o,profiles,wvidx)
objcorr = ReducingGravity.get_correlatedflux(S,objsc)
photometric,corrflux = ReducingGravity.extract_correlated_flux(objcorr)


calpix = ReducingGravity.make_pixels_vector(calib - sky30c.val,profiles,wvidx)
calcorr = ReducingGravity.get_correlatedflux(S,calpix)
photometric,corrflux = ReducingGravity.extract_correlated_flux(calcorr)

bispectra = ReducingGravity.get_bispectrum(corrflux)

using Plots
plotlyjs()
plot!(λ,  mean(abs2.(corrflux[1] ./ sqrt.(max.(1e-2,photometric[1].*photometric[2]))),dims=2); ticks=:native, ylims=[0.,1.], label="1-2r")
plot!(λ, mean(abs2.(corrflux[2] ./ sqrt.(max.(1e-2,photometric[1].*photometric[3]))),dims=2); ticks=:native, ylims=[0.,1.], label="1-3r")
plot!(λ, mean(abs2.(corrflux[3] ./ sqrt.(max.(1e-2,photometric[1].*photometric[4]))),dims=2); ticks=:native, ylims=[0.,1.], label="1-4r")
plot!(λ, mean(abs2.(corrflux[4] ./ sqrt.(max.(1e-2,photometric[2].*photometric[3]))),dims=2); ticks=:native, ylims=[0.,1.], label="2-3r")
plot!(λ, mean(abs2.(corrflux[5] ./ sqrt.(max.(1e-2,photometric[2].*photometric[4]))),dims=2); ticks=:native, ylims=[0.,1.], label="4-2r")
plot!(λ, mean(abs2.(corrflux[6] ./ sqrt.(max.(1e-2,photometric[3].*photometric[4]))),dims=2); ticks=:native, ylims=[0.,1.], label="4-3r")


# plot(λ,  mean(abs2.(corrflux[1]) ./ (4 .*max.(1e-2,photometric[1].*photometric[2])),dims=2); ticks=:native, ylims=[0.,1.], label="1-2")
# plot!(λ, mean(abs2.(corrflux[2]) ./ (4 .*max.(1e-2,photometric[1].*photometric[3])),dims=2); ticks=:native, ylims=[0.,1.], label="1-3")
# plot!(λ, mean(abs2.(corrflux[3]) ./ (4 .*max.(1e-2,photometric[1].*photometric[4])),dims=2); ticks=:native, ylims=[0.,1.], label="1-4")
# plot!(λ, mean(abs2.(corrflux[4]) ./ (4 .*max.(1e-2,photometric[2].*photometric[3])),dims=2); ticks=:native, ylims=[0.,1.], label="2-3")
# plot!(λ, mean(abs2.(corrflux[5]) ./ (4 .*max.(1e-2,photometric[2].*photometric[4])),dims=2); ticks=:native, ylims=[0.,1.], label="4-2")
# plot!(λ, mean(abs2.(corrflux[6]) ./ (4 .*max.(1e-2,photometric[3].*photometric[4])),dims=2); ticks=:native, ylims=[0.,1.], label="4-3")

astrored = openfits("/Users/ferreol/Data/RawData/GRAVITY+/AntoineMerand/reduced_vfactor/GRAVI.2021-10-02T06:41:12.721_astroreduced.fits")
oiflux = read(astrored[8])["FLUX"]
oiwave = read(astrored[2])["EFF_WAVE"]
astrored6 = read(astrored[6])
oidata = astrored6["VISDATA"]
F1F2 = astrored6["F1F2"]
vfactor = astrored6["V_FACTOR"]

plot( oiwave,  mean(abs2.(oidata[:,1:6:end]) ./ (max.(1e-2,F1F2[:,1:6:end] )),dims=2); ticks=:native, ylims=[0.,1.], label="1-2-pip")
plot!( oiwave, mean(abs2.(oidata[:,2:6:end]) ./ (max.(1e-2,F1F2[:,2:6:end] )),dims=2); ticks=:native, ylims=[0.,1.], label="1-3-pip")
plot!( oiwave, mean(abs2.(oidata[:,3:6:end]) ./ (max.(1e-2,F1F2[:,3:6:end] )),dims=2); ticks=:native, ylims=[0.,1.], label="1-4-pip")
plot!( oiwave, mean(abs2.(oidata[:,4:6:end]) ./ (max.(1e-2,F1F2[:,4:6:end] )),dims=2); ticks=:native, ylims=[0.,1.], label="2-3-pip")
plot!( oiwave, mean(abs2.(oidata[:,5:6:end]) ./ (max.(1e-2,F1F2[:,5:6:end] )),dims=2); ticks=:native, ylims=[0.,1.], label="4-2-pip")
plot!( oiwave, mean(abs2.(oidata[:,6:6:end]) ./ (max.(1e-2,F1F2[:,6:6:end] )),dims=2); ticks=:native, ylims=[0.,1.], label="4-3-pip")

#= 
plot( oiwave,  mean(abs.(oidata[:,1:6:end]) ./ sqrt.(max.(1e-2,F1F2[:,1:6:end] .* vfactor[:,1:6:end])),dims=2); ticks=:native, ylims=[0.,1.], label="1-2")
plot!( oiwave, mean(abs.(oidata[:,2:6:end]) ./ sqrt.(max.(1e-2,F1F2[:,2:6:end] .* vfactor[:,2:6:end])),dims=2); ticks=:native, ylims=[0.,1.], label="1-3")
plot!( oiwave, mean(abs.(oidata[:,3:6:end]) ./ sqrt.(max.(1e-2,F1F2[:,3:6:end] .* vfactor[:,3:6:end])),dims=2); ticks=:native, ylims=[0.,1.], label="1-4")
plot!( oiwave, mean(abs.(oidata[:,4:6:end]) ./ sqrt.(max.(1e-2,F1F2[:,4:6:end] .* vfactor[:,4:6:end])),dims=2); ticks=:native, ylims=[0.,1.], label="2-3")
plot!( oiwave, mean(abs.(oidata[:,5:6:end]) ./ sqrt.(max.(1e-2,F1F2[:,5:6:end] .* vfactor[:,5:6:end])),dims=2); ticks=:native, ylims=[0.,1.], label="4-2")
plot!( oiwave, mean(abs.(oidata[:,6:6:end]) ./ sqrt.(max.(1e-2,F1F2[:,6:6:end] .* vfactor[:,6:6:end])),dims=2); ticks=:native, ylims=[0.,1.], label="4-3")
 =#

plot(λ,  .-angle.(corrflux[1][:,1] ); ticks=:native, ylims=[0.,1.], label="1-2r")
plot!(λ, .-angle.(corrflux[2][:,1] ); ticks=:native, ylims=[0.,1.], label="1-3r")
plot!(λ, angle.(corrflux[3][:,1] ); ticks=:native, ylims=[0.,1.], label="4-1r")
plot!(λ, .-angle.(corrflux[4][:,1] ); ticks=:native, ylims=[0.,1.], label="2-3r")
plot!(λ, angle.(corrflux[5][:,1] ); ticks=:native, ylims=[0.,1.], label="4-2r")
plot!(λ, angle.(corrflux[6][:,1] ); ticks=:native, ylims=[0.,1.], label="4-3r")

plot!( oiwave, angle.(oidata[:,1]); ticks=:native, ylims=[0.,1.], label="1-2-pip")
plot!( oiwave, angle.(oidata[:,2]); ticks=:native, ylims=[0.,1.], label="1-3-pip")
plot!( oiwave, angle.(oidata[:,3]); ticks=:native, ylims=[0.,1.], label="1-4-pip")
plot!( oiwave, angle.(oidata[:,4]); ticks=:native, ylims=[0.,1.], label="2-3-pip")
plot!( oiwave, angle.(oidata[:,5]); ticks=:native, ylims=[0.,1.], label="4-2-pip")
plot!( oiwave, angle.(oidata[:,6]); ticks=:native, ylims=[0.,1.], label="4-3-pip")

C = angle.(dropdims(mean(cat(bispectra...,dims=3),dims=2),dims=2))
plot(λ, rad2deg.(C); ticks=:native,xlabel="wavelength",ylabel="Phase [deg]",title="closure phase")