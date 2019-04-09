import SparseArrays.SparseMatrixCSC

abstract type AbstractMesh{Tv} end

struct Mesh{Tv} <: AbstractMesh{Tv}
    adjmat::SparseMatrixCSC{Tv}
    vprop::Matrix{Tv}
    eprop::Matrix{Tv}
    gprop::Vector{Tv}
end