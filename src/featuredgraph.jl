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

const COO_T = Tuple{T, T} where T <: AbstractVector
const ADJMAT_T = AbstractMatrix
const ADJLIST_T = AbstractVector{T} where T <: AbstractVector
# const SPARSE_T = ...  Support sparse adjacency matrices in the future


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

function FeaturedGraph(g; 
                        num_nodes = nothing, 
                        graph_type = :adjmat,
                        dir = :out,
                        nf = nothing, 
                        ef = nothing, 
                        gf = nothing,
                        # ndata = Dict{String, Any}(), 
                        # edata = Dict{String, Any}(),
                        # gdata = Dict{String, Any}()
                        )

    @assert graph_type ∈ [:coo, :adjmat] "Invalid graph_type $graph_type requested"
    @assert dir ∈ [:in, :out]
    if graph_type == :coo
        g, num_nodes, num_edges = to_coo(g; num_nodes, dir)
    else graph_type == :adjmat
        g, num_nodes, num_edges = to_adjmat(g; dir)
    end

    ## Possible future implementation of feature maps. 
    ## Currently this doesn't play well with zygote due to 
    ## https://github.com/FluxML/Zygote.jl/issues/717    
    # ndata["x"] = nf
    # edata["e"] = ef
    # gdata["g"] = gf
    

    FeaturedGraph(g, num_nodes, num_edges, nf, ef, gf)
end

FeaturedGraph(s::AbstractVector, t::AbstractVector; kws...) = FeaturedGraph((s,t); kws...)
FeaturedGraph(g::AbstractGraph; kws...) = FeaturedGraph(adjacency_matrix(g, dir=:out); kws...)

function FeaturedGraph(fg::FeaturedGraph; 
                nf=node_feature(fg), ef=edge_feature(fg), gf=global_feature(fg))
                # ndata=copy(fg.ndata), edata=copy(fg.edata), gdata=copy(fg.gdata), # copy keeps the refs to old data 
    
    FeaturedGraph(fg.graph, fg.num_nodes, fg.num_edges, nf, ef, gf) #   ndata, edata, gdata, 
end


"""
    edge_index(fg::FeaturedGraph)

Return a tuple containing two vectors, respectively containing the source and target 
nodes of the edges in the graph `fg`.

```julia
s, t = edge_index(fg)
```
"""
edge_index(fg::FeaturedGraph{<:COO_T}) = fg.graph

function edge_index(fg::FeaturedGraph{<:ADJMAT_T})
    nz = findall(!=(0), graph(fg)) # vec of cartesian indexes
    ntuple(i -> map(t->t[i], nz), 2)
end

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
LightGraphs.has_vertex(fg::FeaturedGraph, i::Int) = i in 1:fg.num_nodes
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

# TODO return sparse matrix
function LightGraphs.adjacency_matrix(fg::FeaturedGraph{<:COO_T}, T::DataType=Int; dir=:out)
    A, n, m = to_adjmat(fg.graph, T, num_nodes=fg.num_nodes)
    return dir == :out ? A : A'
end

function LightGraphs.adjacency_matrix(fg::FeaturedGraph{<:ADJMAT_T}, T::DataType=eltype(fg.graph); dir=:out)
    @assert dir ∈ [:in, :out]
    A = fg.graph 
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

node_feature(fg::FeaturedGraph) = fg.nf
edge_feature(fg::FeaturedGraph) = fg.ef
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
- `T`: result element type of degree vector; default `Float32`.
- `selfloop`: adding self loop while calculating the matrix.
- `dir`: the edge directionality considered (:out, :in, :both).
"""
function normalized_laplacian(fg::FeaturedGraph, T::DataType=Float32; selfloop::Bool=false, dir::Symbol=:out)
    A = adjacency_matrix(fg, T; dir=dir)
    selfloop && (A += I)
    degs = vec(sum(A; dims=2))
    inv_sqrtD = Diagonal(inv.(sqrt.(degs)))
    return I - inv_sqrtD * A * inv_sqrtD
end

@doc raw"""
    scaled_laplacian(g[, T]; dir=:out)

Scaled Laplacian matrix of graph `g`,
defined as ``\hat{L} = \frac{2}{\lambda_{max}} L - I`` where ``L`` is the normalized Laplacian matrix.

# Arguments

- `g`: should be a adjacency matrix, `FeaturedGraph`, `SimpleGraph`, `SimpleDiGraph` (from LightGraphs) or `SimpleWeightedGraph`, `SimpleWeightedDiGraph` (from SimpleWeightedGraphs).
- `T`: result element type of degree vector; default is the element type of `g` (optional).
- `dir`: the edge directionality considered (:out, :in, :both).
"""
function scaled_laplacian(fg::FeaturedGraph, T::DataType=Float32; dir=:out)
    A = adjacency_matrix(fg, T; dir=dir)
    @assert issymmetric(A) "scaled_laplacian only works with symmetric matrices"
    E = eigen(Symmetric(A)).values
    degs = vec(sum(A; dims=2))
    inv_sqrtD = Diagonal(inv.(sqrt.(degs)))
    Lnorm = I - inv_sqrtD * A * inv_sqrtD
    return  2 / maximum(E) * Lnorm - I
end

function add_self_loops(fg::FeaturedGraph{<:COO_T})
    s, t = edge_index(fg)
    @assert edge_feature(fg) === nothing
    mask_old_loops = s .!= t
    s = s[mask_old_loops]
    t = t[mask_old_loops]
    n = fg.num_nodes
    nodes = convert(typeof(s), [1:n;])
    s = [s; nodes]
    t = [t; nodes]

    FeaturedGraph((s, t), fg.num_nodes, fg.num_edges,
        node_feature(fg), edge_feature(fg), global_feature(fg))
end


function remove_self_loops(fg::FeaturedGraph{<:COO_T})
    s, t = edge_index(fg)
    @assert edge_feature(fg) === nothing
    mask_old_loops = s .!= t
    s = s[mask_old_loops]
    t = t[mask_old_loops]

    FeaturedGraph((s, t), fg.num_nodes, fg.num_edges,
        node_feature(fg), edge_feature(fg), global_feature(fg))
end

@non_differentiable normalized_laplacian(x...)
@non_differentiable scaled_laplacian(x...)
@non_differentiable adjacency_matrix(x...)
@non_differentiable adjacency_list(x...)
@non_differentiable degree(x...)
@non_differentiable add_self_loops(x...)
@non_differentiable remove_self_loops(x...)

# # delete when https://github.com/JuliaDiff/ChainRules.jl/pull/472 is merged
# function ChainRulesCore.rrule(::typeof(copy), x)
#     copy_pullback(ȳ) = (NoTangent(), ȳ)
#     return copy(x), copy_pullback
# end
