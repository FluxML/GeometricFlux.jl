module Datasets
    using DataDeps: DataDep, register, @datadep_str
    using HTTP
    using JLD2
    using JSON
    using LightGraphs: SimpleDiGraph, add_edge!
    using PyCall
    using SparseArrays: SparseMatrixCSC, sparse
    using ZipFile

    export
        Dataset,
        Planetoid,
        Cora,
        PPI,
        Reddit,
        dataset,
        traindata,
        validdata,
        testdata

    include("./dataset.jl")
    include("./planetoid.jl")
    include("./cora.jl")
    include("./ppi.jl")
    include("./reddit.jl")
    # include("./qm7b.jl")
    # include("./entities.jl")
    include("./datautils.jl")

    function __init__()
        planetoid_init()
        cora_init()
        ppi_init()
        reddit_init()
    end
end
