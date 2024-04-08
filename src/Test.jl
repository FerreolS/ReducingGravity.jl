using FITSIO,LinearAlgebra,Statistics, ArrayTools, ImageFiltering,StatsBase
using ReducingGravity


dirpath = "/Users/ferreol/Data/Gravity+/2020-01-06_MEDIUM_COMBINED/"
flist = ReducingGravity.listfitsfiles(dirpath);

Δtflat = first(filter(x -> occursin("FLAT1", x.second.type), flist)).second.Δt

fdark = FITS(first(filter(x -> (occursin(r"(DARK)", x.second.type) && x.second.Δt==Δtflat), flist)).first);
(illuminated,bboxes) = gravi_data_create_bias_mask(fdark);

darkflat = read(fdark["IMAGING_DATA_SC"]);
goodpix = gravi_compute_badpix(darkflat,illuminated)

darkflat,goodpix2  = gravi_create_weighteddata(darkflat,illuminated,goodpix)


flat = Vector{WeightedData{Float32, 2,Matrix{Float32}, Matrix{Float32}}}(undef,4)
goodpixflat = Vector{BitMatrix}(undef,4)

Threads.@threads for i=1:4
	cflat = read(FITS(first(keys(filter(x -> occursin("FLAT$i", x.second.type), flist))))["IMAGING_DATA_SC"]);
	flat[i],goodpixflat[i]  = gravi_create_weighteddata(cflat,illuminated,goodpix)
end

goodpix .&= reduce(.&,goodpixflat)
# if there are update of goodpix map
ReducingGravity.flagbadpix!(darkflat,.!goodpix)
Threads.@threads for i ∈ 1:4
	ReducingGravity.flagbadpix!(flat[i],.!goodpix)
end

profile = gravi_compute_profile(flat .- [darkflat],bboxes,thrsld=0.5)
spctr = gravi_extract_profile_flats(flat .- [darkflat], profile)
lampspectrum = sum(values(spctr)).val ./ length(spctr)


Δtwave = first(filter(x -> occursin(r"(WAVE,LAMP)", x.second.type), flist)).second.Δt

darkwave = read(FITS(first(filter(x -> (occursin(r"(DARK)", x.second.type) && x.second.Δt==Δtwave ), flist)).first)["IMAGING_DATA_SC"]);
darkwave,goodpix  = gravi_create_weighteddata(darkwave,illuminated,goodpix)


fwave = FITS(first(filter(x -> (occursin(r"(WAVE,LAMP)", x.second.type) ), flist)).first);
wave =read(fwave["IMAGING_DATA_SC"]);
wave,goodpix  = gravi_create_weighteddata(wave,illuminated,goodpix)

profile = gravi_spectral_calibration(wave,darkwave, profile)