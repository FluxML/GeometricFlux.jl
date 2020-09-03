using GeometricFlux.Datasets
using SparseArrays: SparseMatrixCSC

@testset "planetoid" begin
    g, train_X, train_y = GeometricFlux.Datasets.traindata(:cora)
    @test typeof(g) == Dict{Any,Any}
    @test typeof(train_X) == SparseMatrixCSC{Float32,Int64}
    @test size(train_X) == (140, 1433)
    @test typeof(train_y) == SparseMatrixCSC{Int32,Int64}
    @test size(train_y) == (140, 7)

    g, test_X, test_y = GeometricFlux.Datasets.testdata(:cora)
    @test typeof(g) == Dict{Any,Any}
    @test typeof(test_X) == SparseMatrixCSC{Float32,Int64}
    @test size(test_X) == (1000, 1433)
    @test typeof(test_y) == SparseMatrixCSC{Int32,Int64}
    @test size(test_y) == (1000, 7)
end