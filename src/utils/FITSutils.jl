import CFITSIO: FITSFile, bitpix_from_type,libcfitsio,fits_assert_ok,fits_update_key,fits_movabs_hdu
#import FITSIO: fits_create_img

name(hdu::HDU) = FITSIO.fits_try_read_extname(hdu.fitsfile)

colnames(hdu::TableHDU) = FITSIO.colnames(hdu)

extver(hdu::HDU) = FITSIO.fits_try_read_extver(hdu.fitsfile)

function getunits(hdr::FITSHeader)
	col_units = Dict{String,String}()
	if !haskey(hdr, "TFIELDS") return nothing
	end
	ncol = hdr["TFIELDS"]
	for i ∈ 1:ncol
		if haskey(hdr, "TUNIT$i") 
			push!(col_units,hdr["TTYPE$i"] => hdr["TUNIT$i"])
		end
	end
	return col_units
end

# function rewind(file::FITSFile)
# 	fits_movabs_hdu(f::FITSFile, 1)
# end

function Base.Dict(hdu::TableHDU) 
	D = Dict{String,Any}()
	for name ∈ colnames(hdu)
		push!(D,name => read(hdu,name))
	end
	return D
end



struct GravityFileInformation
    path::String  # FITS file
    nframes::Int  # number of frames
    Δt::Float64   # exposure time (seconds)
    spectralresolution::String # spectral resolution
    polamode::String #spectral mode
    cat::String   # category name
    type::String  # type name
end

function listfitsfiles( dir::AbstractString = pwd(),
                        suffixes=(".fits", ".fits.gz","fits.Z"))

    FitsDict = Dict{String,GravityFileInformation}();
    for name in readdir(dir)
        path = joinpath(dir, name)
        isfile(path) || continue
        for sfx in suffixes
            if endswith(name, sfx)
                fitshead =  read_header(path)

                caltype=""
                try
                    caltype = fitshead["ESO DPR TYPE"]
                catch
                    continue
                end
                fitshead["ESO INS SHUT11 ST"] ? (caltype  *= "1") :  ()
                fitshead["ESO INS SHUT12 ST"] ? (caltype  *= "2") :  ()
                fitshead["ESO INS SHUT13 ST"] ? (caltype  *= "3") :  ()
                fitshead["ESO INS SHUT14 ST"] ? (caltype  *= "4") :  ()
                calinfo = GravityFileInformation(path,fitshead["ESO DET2 NDIT"],fitshead["ESO DET2 SEQ1 DIT"],fitshead["ESO INS SPEC RES"],fitshead["ESO INS POLA MODE"],fitshead["ESO DPR CATG"],caltype)
                push!(FitsDict, path => calinfo )
                break
            end
        end
    end
    return FitsDict
end

@enum SPECRES LOW MED HIGH
@enum POLAMODE COMBINED SPLIT
@enum CATG CALIB SCIENCE
@enum DPRTYPE DARK FLAT WAVE WAVESC P2VM STDSINGLE SKYSINGLE OBJECTSINGLE

struct GravityData{T,N}
    data::AbstractArray{T,N}  # FITS file
    nframes::Int  # number of frames
    Δt::Float64   # exposure time (seconds)
    spectralresolution::SPECRES # spectral resolution
    polamode::POLAMODE #spectral mode
    cat::CATG   # category name
    type::DPRTYPE  # type name
end
