

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
            if endswith(name, sfx) && !startswith(name,"M.")
                fitshead = read(FitsHeader,path)

                caltype=""
                try
                    caltype = fitshead["ESO DPR TYPE"].string
                catch
                    continue
                end
                fitshead["ESO INS SHUT11 ST"].logical ? (caltype  *= "1") :  ()
                fitshead["ESO INS SHUT12 ST"].logical ? (caltype  *= "2") :  ()
                fitshead["ESO INS SHUT13 ST"].logical ? (caltype  *= "3") :  ()
                fitshead["ESO INS SHUT14 ST"].logical ? (caltype  *= "4") :  ()
                calinfo = GravityFileInformation(path,fitshead["ESO DET2 NDIT"].value,fitshead["ESO DET2 SEQ1 DIT"].value,fitshead["ESO INS SPEC RES"].value,fitshead["ESO INS POLA MODE"].value,fitshead["ESO DPR CATG"].value,caltype)
                push!(FitsDict, path => calinfo )
                break
            end
        end
    end
    return FitsDict
end
