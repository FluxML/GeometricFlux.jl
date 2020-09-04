@testset "cora" begin
    g, X, y = dataset(Cora())
    @test typeof(g) == SparseMatrixCSC{Float32,Int64}
    @test size(g) == (19793, 19793)
    @test typeof(X) == SparseMatrixCSC{Float32,Int64}
    @test size(X) == (19793,  8710)
    @test typeof(y) == Vector{Int64}
    @test size(y) == (19793,)
end