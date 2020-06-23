using CUDAdrv
using GeometricFlux
using Profile
using ProfileView

ENV["JULIA_NUM_THREADS"] = 1

d = 50
nbins = 20
l = 2^20

hist = zeros(Float32, d, nbins)
δ = rand(Float32, d, l)
idx = rand(1:nbins, l)
scatter_add!(hist, δ, idx)

@profile scatter_add!(hist, δ, idx)
Profile.print()

@profview scatter_add!(hist, δ, idx)

# sudo nvprof --profile-from-start off julia benchmark/scatter.jl
# sudo nvprof --profile-from-start off --print-gpu-trace julia --proj benchmark/scatter.jl
# sudo chown $USER -R $HOME/.julia/

# @profview scatter_add!(hist, δ, idx)
# CUDAdrv.@profile scatter_add!(hist_gpu, δ_gpu, idx_gpu)
