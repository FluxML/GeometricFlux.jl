function download_file(url, path)
    HTTP.open("GET", url) do http
        open(path, "w") do file
            write(file, http)
        end
    end
end

function unzip(zipfile::String)
    zipreader = ZipFile.Reader(zipfile)
    dir = dirname(zipfile)
    for f in zipreader.files
        filename = joinpath(dir, f.name)
        open(filename, "w") do file
            write(file, read(f))
        end
    end
    close(zipreader)
end