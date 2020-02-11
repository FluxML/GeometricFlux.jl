using CUDAdrv
using CUDAnative
using CuArrays
using GeometricFlux
using DataFrames
using CSV
using BenchmarkTools
using BenchmarkTools: Trial, TrialEstimate, median, mean
# using ProfileView

d = 50
nbins = 20
getinfo(te::TrialEstimate) = te.time, te.gctime, te.memory
getstats(t::Trial) = [getinfo(minimum(t)), getinfo(mean(t)), getinfo(maximum(t))]

metadata = DataFrame(device=String[], dim=Int[], sample=Int[], bins=Int[])
mintime = DataFrame(min_time=Float64[], min_gc=Float64[], min_mem=Int[])
meantime = DataFrame(mean_time=Float64[], mean_gc=Float64[], mean_mem=Int[])
maxtime = DataFrame(max_time=Float64[], max_gc=Float64[], max_mem=Int[])

for l = [2^5, 2^10, 2^15, 2^20]
    hist = zeros(Float32, d, nbins)
    δ = rand(Float32, d, l)
    idx = rand(1:nbins, l)

    hist_gpu = CuArray(hist)
    δ_gpu = CuArray(δ)
    idx_gpu = CuArray(idx)

    scatter_add!(hist, δ, idx)
    scatter_add!(hist_gpu, δ_gpu, idx_gpu)

    b_cpu = @benchmark scatter_add!($hist, $δ, $idx)
    b_gpu = @benchmark scatter_add!($hist_gpu, $δ_gpu, $idx_gpu)
    s_cpu = getstats(b_cpu)
    s_gpu = getstats(b_gpu)

    push!(metadata, ("cpu", d, l, nbins))
    push!(mintime, s_cpu[1])
    push!(meantime, s_cpu[2])
    push!(maxtime, s_cpu[3])

    push!(metadata, ("gpu", d, l, nbins))
    push!(mintime, s_gpu[1])
    push!(meantime, s_gpu[2])
    push!(maxtime, s_gpu[3])
end

data = hcat(metadata, mintime, meantime, maxtime)
CSV.write("benchmark/scatter_julia.tsv", data; delim="\t")


## Benchmark
# @benchmark scatter_add!($hist, $δ, $idx)
# CuArrays.@time scatter_add!(hist_gpu, δ_gpu, idx_gpu)

## Profiling
# sudo nvprof --profile-from-start off julia benchmark/scatter.jl
# sudo nvprof --profile-from-start off --print-gpu-trace julia --proj benchmark/scatter.jl
# sudo chown yuehhua -R /home/yuehhua/.julia/

# @profview scatter_add!(hist, δ, idx)
# CUDAdrv.@profile scatter_add!(hist_gpu, δ_gpu, idx_gpu)
