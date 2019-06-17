module GeometricFlux
using Requires

include("mesh.jl")
include("layers.jl")

export

    # layers
    # MessagePassing,
    GCNConv,

    ;

function __init__()
    @require MetaGraphs = "626554b9-1ddb-594c-aa3c-2596fe9399a5" begin
        include("metagraphs.jl")
        @require CuArrays = "3a865a2d-5b23-5a0f-bc46-62713ec82fae" include("cuda/metagraphs.jl")
    end

    @require CuArrays = "3a865a2d-5b23-5a0f-bc46-62713ec82fae" begin
        include("cuda/cumesh.jl")
        @require MetaGraphs = "626554b9-1ddb-594c-aa3c-2596fe9399a5" include("cuda/metagraphs.jl")
    end
end

end
