using CSV, DataFrames
using Gadfly
import Cairo

julia_bmk_file = joinpath("benchmark", "scatter_julia.tsv")
python_bmk_file = joinpath("benchmark", "scatter_pytorch.tsv")

bmk_jl = CSV.read(julia_bmk_file; delim='\t')
bmk_py = CSV.read(python_bmk_file; delim='\t')

bmk_jl[!, :framework] .= "geometricflux"
bmk_py[!, :framework] .= "pytorch-scatter"

bmk = vcat(bmk_jl, bmk_py)

bmk[!, :min_time] .= bmk[!, :min_time]/1000
bmk[!, :mean_time] .= bmk[!, :mean_time]/1000
bmk[!, :max_time] .= bmk[!, :max_time]/1000

function plot_benchmark(device)
        DEVICE = uppercase(device)
        p = plot(bmk[bmk[!,:device] .== device, :], x="sample", y="mean_time", color="framework",
                 Geom.point, Geom.line, Scale.x_log2, Scale.y_log10,
                 Guide.title("Scatter add performance on $(DEVICE)"),
                 Guide.xlabel("Matrix Size"), Guide.ylabel("Time (μs)"),
                 Coord.cartesian(xmin=4, xmax=21, ymin=1, ymax=7))

        draw(SVG(joinpath("benchmark", "pics", "$(device)_scatter.svg"), 9inch, 6inch), p)
        draw(PNG(joinpath("benchmark", "pics", "$(device)_scatter.png"), 9inch, 6inch), p)
end

plot_benchmark("cpu")
plot_benchmark("gpu")
