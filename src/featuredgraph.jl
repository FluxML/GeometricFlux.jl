#===================================
Define FeaturedGraph type as a subtype of LightGraphs' AbstractGraph.
For the core methods to be implemented by any AbstractGraph, see
https://juliagraphs.org/LightGraphs.jl/latest/types/#AbstractGraph-Type
https://juliagraphs.org/LightGraphs.jl/latest/developing/#Developing-Alternate-Graph-Types
=============================================#

abstract type AbstractFeaturedGraph <: AbstractGraph{Int} end

"""
    NullGraph()

Null object for `FeaturedGraph`.
"""
struct NullGraph <: AbstractFeaturedGraph end

const COO_T = Tuple{T, T, V} where {T <: AbstractVector,V}
const ADJLIST_T = AbstractVector{T} where T <: AbstractVector
const ADJMAT_T = AbstractMatrix
const SPARSE_T = AbstractSparseMatrix # subset of ADJMAT_T

"""
    FeaturedGraph(g; [graph_type, dir, num_nodes, nf, ef, gf])
    FeaturedGraph(fg::FeaturedGraph; [nf, ef, gf])

A type representing a graph structure and storing also arrays 
that contain features associated to nodes, edges, and the whole graph. 
    
A `FeaturedGraph` can be constructed out of different objects `g` representing
the connections inside the graph, while the internal representation type
is governed by `graph_type`. 
When constructed from another featured graph `fg`, the internal graph representation
is preserved and shared. 

A `FeaturedGraph` is a LightGraphs' `AbstractGraph`, therefore any functionality
from the LightGraphs' graph library can be used on it.

# Arguments 

- `g`: Some data representing the graph topology. Possible type are 
    - An adjacency matrix
    - An adjacency list.
    - A tuple containing the source and target vectors (COO representation)
    - A LightGraphs' graph.
- `graph_type`: A keyword argument that specifies 
                the underlying representation used by the FeaturedGraph. 
                Currently supported values are 
    - `:coo`. Graph represented as a tuple `(source, target)`, such that the `k`-th edge 
              connects the node `source[k]` to node `target[k]`.
              Optionally, also edge weights can be given: `(source, target, weights)`.
    - `:sparse`. A sparse adjacency matrix representation.
    - `:dense`. A dense adjacency matrix representation.  
    Default `:coo`.
- `dir`. The assumed edge direction when given adjacency matrix or adjacency list input data `g`. 
        Possible values are `:out` and `:in`. Defaul `:out`.
- `num_nodes`. The number of nodes. If not specified, inferred from `g`. Default nothing.
- `nf`: Node features. Either nothing, or an array whose last dimension has size num_nodes. Default nothing.
- `ef`: Edge features. Either nothing, or an array whose last dimension has size num_edges. Default nothing.
- `gf`: Global features. Default nothing. 

# Usage. 

```
using Flux, GeometricFlux

# Construct from adjacency list representation
g = [[2,3], [1,4,5], [1], [2,5], [2,4]]
fg = FeaturedGraph(g)

# Number of nodes and edges
fg.num_nodes  # 5
fg.num_edges  # 10 

# Same graph in COO representation
s = [1,1,2,2,2,3,4,4,5,5]
t = [2,3,1,4,5,3,2,5,2,4]
fg = FeaturedGraph((s, t))
fg = FeaturedGraph(s, t) # other convenience constructor

# From a LightGraphs' graph
fg = FeaturedGraph(erdos_renyi(100, 20))

# Copy featured graph while also adding node features
fg = FeaturedGraph(fg, nf=rand(100, 5))

# Send to gpu
fg = fg |> gpu

# Collect edges' source and target nodes.
# Both source and target are vectors of length num_edges
source, target = edge_index(fg)
```

See also [`graph`](@ref), [`edge_index`](@ref), [`node_feature`](@ref), [`edge_feature`](@ref), and [`global_feature`](@ref) 
"""
struct FeaturedGraph{T<:Union{COO_T,ADJMAT_T}} <: AbstractFeaturedGraph
    graph::T
    num_nodes::Int
    num_edges::Int
    nf
    ef
    gf
    ## possible future property stores
    # ndata::Dict{String, Any} # https://github.com/FluxML/Zygote.jl/issues/717        
    # edata::Dict{String, Any}
    # gdata::Dict{String, Any}
end

@functor FeaturedGraph

function FeaturedGraph(data; 
                        num_nodes = nothing, 
                        graph_type = :coo,
                        dir = :out,
                        nf = nothing, 
                        ef = nothing, 
                        gf = nothing,
                        # ndata = Dict{String, Any}(), 
                        # edata = Dict{String, Any}(),
                        # gdata = Dict{String, Any}()
                        )

    @assert graph_type ∈ [:coo, :dense, :sparse] "Invalid graph_type $graph_type requested"
    @assert dir ∈ [:in, :out]
    if graph_type == :coo
        g, num_nodes, num_edges = to_coo(data; num_nodes, dir)
    elseif graph_type == :dense
        g, num_nodes, num_edges = to_dense(data; dir)
    elseif graph_type == :sparse
        g, num_nodes, num_edges = to_sparse(data; dir)
    end

    ## Possible future implementation of feature maps. 
    ## Currently this doesn't play well with zygote due to 
    ## https://github.com/FluxML/Zygote.jl/issues/717    
    # ndata["x"] = nf
    # edata["e"] = ef
    # gdata["g"] = gf
    

    FeaturedGraph(g, num_nodes, num_edges, nf, ef, gf)
end

# COO convenience constructors
FeaturedGraph(s::AbstractVector, t::AbstractVector, v = nothing; kws...) = FeaturedGraph((s, t, v); kws...)
FeaturedGraph((s, t)::NTuple{2}; kws...) = FeaturedGraph((s, t, nothing); kws...)

# FeaturedGraph(g::AbstractGraph; kws...) = FeaturedGraph(adjacency_matrix(g, dir=:out); kws...)

function FeaturedGraph(g::AbstractGraph; kws...)
    s = LightGraphs.src.(LightGraphs.edges(g))
    t = LightGraphs.dst.(LightGraphs.edges(g)) 
    FeaturedGraph((s, t); kws...)
end

function FeaturedGraph(fg::FeaturedGraph; 
                nf=node_feature(fg), ef=edge_feature(fg), gf=global_feature(fg))
                # ndata=copy(fg.ndata), edata=copy(fg.edata), gdata=copy(fg.gdata), # copy keeps the refs to old data 
    
    FeaturedGraph(fg.graph, fg.num_nodes, fg.num_edges, nf, ef, gf) #   ndata, edata, gdata, 
end


"""
    edge_index(fg::FeaturedGraph)

Return a tuple containing two vectors, respectively storing 
the source and target nodes for each edges in `fg`.

```julia
s, t = edge_index(fg)
```
"""
edge_index(fg::FeaturedGraph{<:COO_T}) = graph(fg)[1:2]

edge_index(fg::FeaturedGraph{<:ADJMAT_T}) = to_coo(graph(fg))[1][1:2]

edge_weight(fg::FeaturedGraph{<:COO_T}) = graph(fg)[3]

"""
    graph(fg::FeaturedGraph)

Return the underlying implementation of the graph structure of `fg`,
either an adjacency matrix or an edge list in the COO format.
"""
graph(fg::FeaturedGraph) = fg.graph

LightGraphs.edges(fg::FeaturedGraph) = zip(edge_index(fg)...)

LightGraphs.edgetype(fg::FeaturedGraph) = Tuple{Int, Int}

function LightGraphs.has_edge(fg::FeaturedGraph{<:COO_T}, i::Integer, j::Integer)
    s, t = edge_index(fg)
    return any((s .== i) .& (t .== j))
end

LightGraphs.has_edge(fg::FeaturedGraph{<:ADJMAT_T}, i::Integer, j::Integer) = graph(fg)[i,j] != 0

LightGraphs.nv(fg::FeaturedGraph) = fg.num_nodes
LightGraphs.ne(fg::FeaturedGraph) = fg.num_edges
LightGraphs.has_vertex(fg::FeaturedGraph, i::Int) = 1 <= i <= fg.num_nodes
LightGraphs.vertices(fg::FeaturedGraph) = 1:fg.num_nodes

function LightGraphs.outneighbors(fg::FeaturedGraph{<:COO_T}, i::Integer)
    s, t = edge_index(fg)
    return t[s .== i]
end

function LightGraphs.outneighbors(fg::FeaturedGraph{<:ADJMAT_T}, i::Integer)
    A = graph(fg)
    return findall(!=(0), A[i,:])
end

function LightGraphs.inneighbors(fg::FeaturedGraph{<:COO_T}, i::Integer)
    s, t = edge_index(fg)
    return s[t .== i]
end

function LightGraphs.inneighbors(fg::FeaturedGraph{<:ADJMAT_T}, i::Integer)
    A = graph(fg)
    return findall(!=(0), A[:,i])
end

LightGraphs.is_directed(::FeaturedGraph) = true
LightGraphs.is_directed(::Type{FeaturedGraph}) = true

function adjacency_list(fg::FeaturedGraph; dir=:out)
    @assert dir ∈ [:out, :in]
    fneighs = dir == :out ? outneighbors : inneighbors
    return [fneighs(fg, i) for i in 1:fg.num_nodes]
end

function LightGraphs.adjacency_matrix(fg::FeaturedGraph{<:COO_T}, T::DataType=Int; dir=:out)
    A, n, m = to_sparse(graph(fg), T, num_nodes=fg.num_nodes)
    @assert size(A) == (n, n)
    return dir == :out ? A : A'
end

function LightGraphs.adjacency_matrix(fg::FeaturedGraph{<:ADJMAT_T}, T::DataType=eltype(graph(fg)); dir=:out)
    @assert dir ∈ [:in, :out]
    A = graph(fg) 
    A = T != eltype(A) ? T.(A) : A
    return dir == :out ? A : A'
end

function LightGraphs.degree(fg::FeaturedGraph{<:COO_T}; dir=:out)
    s, t = edge_index(fg)
    degs = fill!(similar(s, eltype(s), fg.num_nodes), 0)
    o = fill!(similar(s, eltype(s), fg.num_edges), 1)
    if dir ∈ [:out, :both]
        NNlib.scatter!(+, degs, o, s)
    end
    if dir ∈ [:in, :both]
        NNlib.scatter!(+, degs, o, t)
    end
    return degs
end

function LightGraphs.degree(fg::FeaturedGraph{<:ADJMAT_T}; dir=:out)
    @assert dir ∈ (:in, :out)
    A = graph(fg)
    return dir == :out ? vec(sum(A, dims=2)) : vec(sum(A, dims=1))
end

# node_feature(fg::FeaturedGraph) = fg.ndata["x"]
# edge_feature(fg::FeaturedGraph) = fg.edata["e"]
# global_feature(fg::FeaturedGraph) = fg.gdata["g"]


"""
    node_feature(fg::FeaturedGraph)

Return the node features of `fg`.
"""
node_feature(fg::FeaturedGraph) = fg.nf

"""
    edge_feature(fg::FeaturedGraph)

Return the edge features of `fg`.
"""
edge_feature(fg::FeaturedGraph) = fg.ef

"""
    global_feature(fg::FeaturedGraph)

Return the global features of `fg`.
"""
global_feature(fg::FeaturedGraph) = fg.gf

# function Base.getproperty(fg::FeaturedGraph, sym::Symbol)
#     if sym === :nf
#         return fg.ndata["x"]
#     elseif sym === :ef
#         return fg.edata["e"]
#     elseif sym === :gf
#         return fg.gdata["g"]
#     else # fallback to getfield
#         return getfield(fg, sym)
#     end
# end

function LightGraphs.laplacian_matrix(fg::FeaturedGraph, T::DataType=Int; dir::Symbol=:out)
    A = adjacency_matrix(fg, T; dir=dir)
    D = Diagonal(vec(sum(A; dims=2)))
    return D - A
end

"""
    normalized_laplacian(fg, T=Float32; selfloop=false, dir=:out)

Normalized Laplacian matrix of graph `g`.

# Arguments

- `fg`: A `FeaturedGraph`.
- `T`: result element type.
- `selfloop`: adding self loop while calculating the matrix.
- `dir`: the edge directionality considered (:out, :in, :both).
"""
function normalized_laplacian(fg::FeaturedGraph, T::DataType=Float32; selfloop::Bool=false, dir::Symbol=:out)
    A = adjacency_matrix(fg, T; dir=dir)
    sz = size(A)
    @assert sz[1] == sz[2]
    if selfloop
        A += I - Diagonal(A)
    else
        A -= Diagonal(A) 
    end
    degs = vec(sum(A; dims=2))
    inv_sqrtD = Diagonal(inv.(sqrt.(degs)))
    return I - inv_sqrtD * A * inv_sqrtD
end

@doc raw"""
    scaled_laplacian(fg, T=Float32; dir=:out)

Scaled Laplacian matrix of graph `g`,
defined as ``\hat{L} = \frac{2}{\lambda_{max}} L - I`` where ``L`` is the normalized Laplacian matrix.

# Arguments

- `fg`: A `FeaturedGraph`.
- `T`: result element type.
- `dir`: the edge directionality considered (:out, :in, :both).
"""
function scaled_laplacian(fg::FeaturedGraph, T::DataType=Float32; dir=:out)
    L = normalized_laplacian(fg, T)
    @assert issymmetric(L) "scaled_laplacian only works with symmetric matrices"
    λmax = _eigmax(L)
    return  2 / λmax * L - I
end

# _eigmax(A) = eigmax(Symmetric(A)) # Doesn't work on sparse arrays
_eigmax(A) = KrylovKit.eigsolve(Symmetric(A), 1, :LR)[1][1] # also eigs(A, x0, nev, mode) available 

# Eigenvalues for cuarray don't seem to be well supported. 
# https://github.com/JuliaGPU/CUDA.jl/issues/154
# https://discourse.julialang.org/t/cuda-eigenvalues-of-a-sparse-matrix/46851/5

"""
    add_self_loops(fg::FeaturedGraph)

Return a featured graph with the same features as `fg`
but also adding edges connecting the nodes to themselves.
"""
function add_self_loops(fg::FeaturedGraph{<:COO_T})
    s, t = edge_index(fg)
    @assert edge_feature(fg) === nothing
    @assert edge_weight(fg) === nothing
    mask_old_loops = s .!= t
    s = s[mask_old_loops]
    t = t[mask_old_loops]
    n = fg.num_nodes
    nodes = convert(typeof(s), [1:n;])
    s = [s; nodes]
    t = [t; nodes]

    FeaturedGraph((s, t, nothing), fg.num_nodes, length(s),
        node_feature(fg), edge_feature(fg), global_feature(fg))
end

function add_self_loops(fg::FeaturedGraph{<:ADJMAT_T})
    A = graph(fg)
    @assert edge_feature(fg) === nothing
    nold = sum(Diagonal(A)) |> Int
    A = A - Diagonal(A) + I
    num_edges =  fg.num_edges - nold + fg.num_nodes
    FeaturedGraph(A, fg.num_nodes, num_edges,
        node_feature(fg), edge_feature(fg), global_feature(fg))
end


function remove_self_loops(fg::FeaturedGraph{<:COO_T})
    s, t = edge_index(fg)
    # TODO remove these constraints
    @assert edge_feature(fg) === nothing
    @assert edge_weight(fg) === nothing
    
    mask_old_loops = s .!= t
    s = s[mask_old_loops]
    t = t[mask_old_loops]

    FeaturedGraph((s, t, nothing), fg.num_nodes, length(s),
        node_feature(fg), edge_feature(fg), global_feature(fg))
end

@non_differentiable normalized_laplacian(x...)
@non_differentiable scaled_laplacian(x...)
@non_differentiable adjacency_matrix(x...)
@non_differentiable adjacency_list(x...)
@non_differentiable degree(x...)
@non_differentiable add_self_loops(x...)     # TODO this is wrong, since fg carries feature arrays, needs rrule
@non_differentiable remove_self_loops(x...)  # TODO this is wrong, since fg carries feature arrays, needs rrule

# # delete when https://github.com/JuliaDiff/ChainRules.jl/pull/472 is merged
# function ChainRulesCore.rrule(::typeof(copy), x)
#     copy_pullback(ȳ) = (NoTangent(), ȳ)
#     return copy(x), copy_pullback
# end
