using CuArrays
using CuArrays.CUSPARSE

struct CuMesh{Tv} <: AbstractMesh{Tv}
    adjmat::CuSparseMatrixCSC{Tv}
    vprop::CuMatrix{Tv}
    eprop::CuMatrix{Tv}
    gprop::CuVector{Tv}
end
