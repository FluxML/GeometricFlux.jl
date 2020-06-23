using PyCall
using DataFrames
using CSV
using BenchmarkTools
using BenchmarkTools: Trial, TrialEstimate, median, mean

py"""
import torch
import torch_scatter as sc
cuda = torch.device("cuda:0")
d = 50
nbins = 20
"""

d = 50
nbins = 20
getinfo(te::TrialEstimate) = te.time, te.gctime, te.memory
getstats(t::Trial) = [getinfo(minimum(t)), getinfo(mean(t)), getinfo(maximum(t))]

metadata = DataFrame(device=String[], dim=Int[], sample=Int[], bins=Int[])
mintime = DataFrame(min_time=Float64[], min_gc=Float64[], min_mem=Int[])
meantime = DataFrame(mean_time=Float64[], mean_gc=Float64[], mean_mem=Int[])
maxtime = DataFrame(max_time=Float64[], max_gc=Float64[], max_mem=Int[])

for l = [2^5, 2^10, 2^15, 2^20]
    py"""
    hist = torch.zeros([d, nbins], dtype=torch.float32)
    delta = torch.rand([d, $(l)], dtype=torch.float32)
    idx = torch.randint(0, nbins, size=($(l),))

    hist_gpu = torch.zeros([d, nbins], dtype=torch.float32, device=cuda)
    delta_gpu = torch.rand([d, $(l)], dtype=torch.float32, device=cuda)
    idx_gpu = torch.randint(0, nbins, size=($(l),), device=cuda)

    sc.scatter_add(delta, idx, out=hist)
    sc.scatter_add(delta_gpu, idx_gpu, out=hist_gpu)
    """

    b_cpu = @benchmark py"sc.scatter_add(delta, idx, out=hist)";
    b_gpu = @benchmark py"sc.scatter_add(delta_gpu, idx_gpu, out=hist_gpu)";
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
CSV.write("benchmark/scatter_pytorch.tsv", data; delim="\t")
