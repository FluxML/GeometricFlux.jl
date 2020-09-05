@testset "ppi" begin
    g, train_X, train_y, train_ids = traindata(PPI())
    @test typeof(g) == SimpleDiGraph{Int32}
    @test nv(g) == 44906
    @test ne(g) == 1271267
    @test typeof(train_X) == Array{Float32,2}
    @test size(train_X) == (44906, 50)
    @test typeof(train_y) == SparseMatrixCSC{Int32,Int64}
    @test size(train_y) == (44906, 121)

    g, valid_X, valid_y, valid_ids = validdata(PPI())
    @test typeof(g) == SimpleDiGraph{Int32}
    @test nv(g) == 6514
    @test ne(g) == 205395
    @test typeof(valid_X) == Array{Float32,2}
    @test size(valid_X) == (6514, 50)
    @test typeof(valid_y) == SparseMatrixCSC{Int32,Int64}
    @test size(valid_y) == (6514, 121)

    g, test_X, test_y, test_ids = testdata(PPI())
    @test typeof(g) == SimpleDiGraph{Int32}
    @test nv(g) == 5524
    @test ne(g) == 167461
    @test typeof(test_X) == Array{Float32,2}
    @test size(test_X) == (5524, 50)
    @test typeof(test_y) == SparseMatrixCSC{Int32,Int64}
    @test size(test_y) == (5524, 121)
end