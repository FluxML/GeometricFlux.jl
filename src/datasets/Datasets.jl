module Datasets
    using DataDeps: DataDep, register, @datadep_str
    using HTTP
    using JLD2
    using NPZ: npzread
    using PyCall
    using SparseArrays: SparseMatrixCSC

    export
        Dataset,
        Planetoid,
        traindata,
        testdata

    abstract type Dataset end

    include("./planetoid.jl")
    include("./cora.jl")
    include("./ppi.jl")
    include("./reddit.jl")
    # include("./qm7b.jl")
    # include("./entities.jl")
    include("./datautils.jl")

    function __init__()
        planetoid_init()
    end
end
