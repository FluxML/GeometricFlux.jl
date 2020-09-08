function download_file(url, path)
    HTTP.open("GET", url) do http
        open(path, "w") do file
            write(file, http)
        end
    end
end

function unzip(zipfile::String)
    f = replace(zipfile, ".zip"=>"")
    run(`unzip $f`)
end