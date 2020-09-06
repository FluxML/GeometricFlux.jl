@testset "reddit" begin
    g, X, y, ids, types = dataset(Reddit())
    @test typeof(g) == SparseMatrixCSC{Int32,Int64}
    @test size(g) == (232965, 232965)
    @test typeof(X) == Matrix{Float32}
    @test size(X) == (232965, 602)
    @test typeof(y) == Vector{Int32}
    @test size(y) == (232965,)
    @test typeof(ids) == Vector{Int32}
    @test size(ids) == (232965,)
    @test typeof(types) == Vector{Int32}
    @test size(types) == (232965,)
end