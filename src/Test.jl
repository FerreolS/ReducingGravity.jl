using FITSIO,Statistics, ArrayTools,StatsBase
using ReducingGravity


dirpath = "/Users/ferreol/Data/Gravity+/2020-01-06_MEDIUM_COMBINED/"
flist = ReducingGravity.listfitsfiles(dirpath);

Δtflat = first(filter(x -> occursin("FLAT1", x.second.type), flist)).second.Δt

fdark = FITS(first(filter(x -> (occursin(r"(DARK)", x.second.type) && x.second.Δt==Δtflat), flist)).first);
(illuminated,bboxes) = gravi_data_create_bias_mask(fdark);

darkflat = read(fdark["IMAGING_DATA_SC"]);
goodpix = gravi_compute_badpix(darkflat,illuminated,spatialkernel=(11,1))
illuminated = illuminated .|| .!goodpix


darkflat,goodpix  = gravi_create_weighteddata(darkflat,illuminated,goodpix)


flat = Vector{WeightedData{Float32, 2,Matrix{Float32}, Matrix{Float32}}}(undef,4)
cflat = Vector{Array{Float32,3}}(undef,4)
goodpixflat = Vector{BitMatrix}(undef,4)

Threads.@threads for i=1:4
	cflat[i] = read(FITS(first(keys(filter(x -> occursin("FLAT$i", x.second.type), flist))))["IMAGING_DATA_SC"]);
	flat[i],goodpixflat[i]  = gravi_create_weighteddata(cflat[i],illuminated,goodpix)
end

goodpix .&= reduce(.&,goodpixflat)
# if there are update of goodpix map
flagbadpix!(darkflat,.!goodpix)
Threads.@threads for i ∈ 1:4
	flagbadpix!(flat[i],.!goodpix)
end

profiles = gravi_compute_profile(flat .- [darkflat],bboxes,thrsld=0.1)


Δtwave = first(filter(x -> occursin(r"(WAVE,LAMP)", x.second.type), flist)).second.Δt

darkwave = read(FITS(first(filter(x -> (occursin(r"(DARK)", x.second.type) && x.second.Δt==Δtwave ), flist)).first)["IMAGING_DATA_SC"]);
darkwave,goodpix  = gravi_create_weighteddata(darkwave,illuminated,goodpix)


fwave = FITS(first(filter(x -> (occursin(r"(WAVE,LAMP)", x.second.type) ), flist)).first);
wave =read(fwave["IMAGING_DATA_SC"]);
wave,goodpix  = gravi_create_weighteddata(wave,illuminated,goodpix)

profiles = gravi_spectral_calibration(wave,darkwave, profiles; nonnegative=true, robust=true)



len_p2vm = length(filter(x -> occursin("P2VM", x.second.type), flist))
P2VM = Vector{Pair{String, Array{Float32, 3}}}(undef,len_p2vm)
P2VMwd = Vector{Pair{String,ConcreteWeightedData{Float32,2}}}(undef,len_p2vm)
Threads.@threads for (i,pv2mfile) ∈ collect(enumerate(values(filter(x -> occursin("P2VM", x.second.type), flist))))
	rawdata =read(FITS(pv2mfile.path)["IMAGING_DATA_SC"])
	P2VM[i] = pv2mfile.type => gravi_data_detector_cleanup(rawdata,illuminated,keepbias=true)[2]
	data,_ = gravi_create_weighteddata(rawdata,illuminated,goodpix; filterblink=true, blinkkernel=(1,1,9),keepbias=true, cleanup=false)
	P2VMwd[i] = pv2mfile.type =>data
end

P2VM = Dict(P2VM)
P2VMwd = Dict(P2VMwd)

darkp2vm, gain, rov = gravi_compute_gain_from_p2vm(P2VMwd,profiles,goodpix)
spctr = gravi_extract_profile_flats_from_p2vm(P2VMwd , darkp2vm,profiles; nonnegative=true, robust=false)

profiles, lamp = gravi_compute_lamp_transmissions(  spctr, profiles)#,kernel = InterpolationKernels.BSpline{1}())

#wd = gravi_create_weighteddata(P2VM["P2VM12"], illuminated,goodpix,rov, gain)
wd = gravi_create_weighteddata(P2VM["P2VM12"], illuminated,goodpix,rov, gain)
A = ReducingGravity.gravi_extract_channel(wd-darkp2vm,profiles["12-A-C"],lamp)
B = ReducingGravity.gravi_extract_channel(wd-darkp2vm,profiles["12-B-C"],lamp)
C = ReducingGravity.gravi_extract_channel(wd-darkp2vm,profiles["12-C-C"],lamp)
D = ReducingGravity.gravi_extract_channel(wd-darkp2vm,profiles["12-D-C"],lamp)
ϕ = ReducingGravity.gravi_initial_input_phase(A,B,C,D)
phasors= ReducingGravity.gravi_build_ABCD_phasors(ϕ,A,B,C,D);
phase = ReducingGravity.estimate_visibility(phasors,A,B,C,D);
rho = sqrt.(phase[:,:,1].^2 .+ phase[:,:,2] .^2)
#rho3 = (ones(360) .* median(rho,dims=1))
phase .*= 1 ./ rho # .* rho3
phasors= ReducingGravity.gravi_build_ABCD_phasors(phase,A,B,C,D);
phase = ReducingGravity.estimate_visibility(phasors,A,B,C,D);