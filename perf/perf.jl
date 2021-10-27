using Flux, GeometricFlux, Graphs, BenchmarkTools, CUDA
using DataFrames, Statistics, JLD2, SparseArrays

BenchmarkTools.ratio(::Missing, x) = Inf
BenchmarkTools.ratio(x, ::Missing) = 0.0
BenchmarkTools.ratio(::Missing, ::Missing) = missing

adjlist(g) = [neighbors(g, i) for i in 1:nv(g)]

function run_single_benchmark(N, c, D, CONV; gtype=:lg)
    @assert gtype ∈ [:lightgraph, :adjlist, :dense, :sparse]
    g = erdos_renyi(N, c / (N-1), seed=17)
    if gtype == :adjlist
        g = adjlist(g)
    elseif gtype == :dense
        g = Array(adjacency_matrix(g))
    elseif gtype == :sparse 
        g = adjacency_matrix(g) # lightgraphs returns sparse adj mats
    end

    res = Dict() 
    X = randn(Float32, D, N)
    res["FG"] = @benchmark FeaturedGraph($g, nf=$X) 
    
    fg = FeaturedGraph(g, nf=X)
    fg_gpu = fg |> gpu    
    
    m = CONV(D => D)
    m_gpu = m |> gpu
    try 
        res["CPU"] = @benchmark $m($fg)
    catch
        res["CPU"] = missing
    end

    try 
        res["GPU"] = @benchmark CUDA.@sync($m_gpu($fg_gpu)) teardown=(GC.gc(); CUDA.reclaim())
    catch
        res["GPU"] = missing
    end

    return res
end
"""
    run_benchmarks(;
        Ns = [10, 100, 1000, 10000],
        c = 6,
        D = 100)

Benchmark GNN layers on random regular graphs 
of mean connectivity `c` and number of nodes in the list `Ns`.
`D` is the number of node features.
"""
function run_benchmarks(; 
        Ns = [10, 100, 1000, 10000],
        c = 6.0,
        D = 100)

    df = DataFrame(N=Int[], c=Float64[], layer=String[], gtype=Symbol[], 
                   time_fg=Any[], time_cpu=Any[], time_gpu=Any[]) |> allowmissing
    
    for gtype in [:lightgraph, :adjlist, :dense, :sparse]
        for N in Ns
            println("## GRAPH_TYPE = $gtype  N = $N")           
            for CONV in [GCNConv, GraphConv, GATConv]
                res = run_single_benchmark(N, c, D, CONV; gtype)
                row = (; N = N,
                        c = c,
                        layer = "$CONV", 
                        gtype = gtype, 
                        time_fg = median(res["FG"]),
                        time_cpu = ismissing(res["CPU"]) ? missing : median(res["CPU"]),
                        time_gpu = ismissing(res["GPU"]) ? missing : median(res["GPU"]),
                    )
                push!(df, row)
            end
        end
    end
    df.gpu_to_cpu = ratio.(df.time_gpu, df.time_cpu)
    sort!(df, [:layer, :N, :c, :gtype])
    return df
end

# df = run_benchmarks()
# for g in groupby(df, :layer); println(g, "\n"); end

# @save "perf/perf_master_20210803_carlo.jld2" dfmaster=df
## or
# @save "perf/perf_pr.jld2" dfpr=df


function compare(dfpr, dfmaster; on=[:N, :c, :layer])
    df = outerjoin(dfpr, dfmaster; on=on, makeunique=true, renamecols = :_pr => :_master)
    df.pr_to_master_cpu = ratio.(df.time_cpu_pr, df.time_cpu_master)
    df.pr_to_master_gpu = ratio.(df.time_cpu_pr, df.time_gpu_master) 
    return df[:,[:N, :c, :gtype_pr, :gtype_master, :layer, :pr_to_master_cpu, :pr_to_master_gpu]]
end

# @load "perf/perf_pr.jld2" dfpr
# @load "perf/perf_master.jld2" dfmaster
# compare(dfpr, dfmaster)
