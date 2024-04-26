using FITSIO,LinearAlgebra,Statistics, ArrayTools, ImageFiltering,StatsBase
using ReducingGravity


dirpath = "/Users/ferreol/Data/Gravity+/2020-01-06_MEDIUM_COMBINED/"
flist = ReducingGravity.listfitsfiles(dirpath);

Δtflat = first(filter(x -> occursin("FLAT1", x.second.type), flist)).second.Δt

fdark = FITS(first(filter(x -> (occursin(r"(DARK)", x.second.type) && x.second.Δt==Δtflat), flist)).first);
(illuminated,bboxes) = gravi_data_create_bias_mask(fdark);

darkflat = read(fdark["IMAGING_DATA_SC"]);
goodpix = gravi_compute_badpix(darkflat,illuminated)
illuminated = illuminated .|| .!goodpix


darkflat,goodpix2  = gravi_create_weighteddata(darkflat,illuminated,goodpix)


flat = Vector{WeightedData{Float32, 2,Matrix{Float32}, Matrix{Float32}}}(undef,4)
cflat = Vector{Array{Float32,3}}(undef,4)
goodpixflat = Vector{BitMatrix}(undef,4)

Threads.@threads for i=1:4
	cflat[i] = read(FITS(first(keys(filter(x -> occursin("FLAT$i", x.second.type), flist))))["IMAGING_DATA_SC"]);
	flat[i],goodpixflat[i]  = gravi_create_weighteddata(cflat[i],illuminated,goodpix)
end

goodpix .&= reduce(.&,goodpixflat)
# if there are update of goodpix map
ReducingGravity.flagbadpix!(darkflat,.!goodpix)
Threads.@threads for i ∈ 1:4
	ReducingGravity.flagbadpix!(flat[i],.!goodpix)
end

profiles = gravi_compute_profile(flat .- [darkflat],bboxes,thrsld=0.5)
#spctr = gravi_extract_profile_flats(flat .- [darkflat], profiles)
#ron,gain = gravi_compute_gain(cflat,illuminated,goodpix,profiles)
#lampspectrum = sum(values(spctr)).val ./ length(spctr)
#trans,lamp = gravi_compute_transmission(spctr; maxeval=50)
#profiles,lamp = gravi_compute_transmissions(spctr,profiles)


Δtwave = first(filter(x -> occursin(r"(WAVE,LAMP)", x.second.type), flist)).second.Δt

darkwave = read(FITS(first(filter(x -> (occursin(r"(DARK)", x.second.type) && x.second.Δt==Δtwave ), flist)).first)["IMAGING_DATA_SC"]);
darkwave,goodpix  = gravi_create_weighteddata(darkwave,illuminated,goodpix)


fwave = FITS(first(filter(x -> (occursin(r"(WAVE,LAMP)", x.second.type) ), flist)).first);
wave =read(fwave["IMAGING_DATA_SC"]);
wave,goodpix  = gravi_create_weighteddata(wave,illuminated,goodpix)

wav = gravi_extract_profile(wave - darkwave, profiles)
profiles = gravi_spectral_calibration(wave,darkwave, profiles)



len_p2vm = length(filter(x -> occursin("P2VM", x.second.type), flist))
P2VM = Vector{Pair{String, Array{Float32, 3}}}(undef,len_p2vm)
P2VMwd = Vector{Pair{String,AbstractWeightedData{Float32,2}}}(undef,len_p2vm)
Threads.@threads for (i,pv2mfile) ∈ collect(enumerate(values(filter(x -> occursin("P2VM", x.second.type), flist))))
	rawdata =read(FITS(pv2mfile.path)["IMAGING_DATA_SC"])
	P2VM[i] = pv2mfile.type =>rawdata
	data,_ = gravi_create_weighteddata(rawdata,illuminated,goodpix; filterblink=true, blinkkernel=(1,1,9),keepbias=true)
	P2VMwd[i] = pv2mfile.type =>data
end

P2VM = Dict(P2VM)
P2VMwd = Dict(P2VMwd)

spctr = gravi_extract_profile_flats_from_p2vm(P2VMwd , darkflat,profiles)
profiles,lamp = gravi_compute_transmissions(spctr,profiles)
gain, ron = gravi_compute_gain_from_p2vm(P2VMwd,profiles,goodpix)