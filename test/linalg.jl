@testset "Test Linear Algebra" begin
    adj = [0 1 0 1;
           1 0 1 0;
           0 1 0 1;
           1 0 1 0]
    deg = [2 0 0 0;
           0 2 0 0;
           0 0 2 0;
           0 0 0 2]
    lap = [2 -1 0 -1;
           -1 2 -1 0;
           0 -1 2 -1;
           -1 0 -1 2]
    norm_lap = [1. -.5 0. -.5;
               -.5 1. -.5 0.;
               0. -.5 1. -.5;
               -.5 0. -.5 1.]

    for T in [Int8, Float64]
        @test degree_matrix(adj, T, dir=:out) == T.(deg)
        @test degree_matrix(adj, T, dir=:out) == degree_matrix(adj, T, dir=:in)
        @test laplacian_matrix(adj, T) == T.(lap)
    end
    @test normalized_laplacian(adj, Float64) ≈ norm_lap
    @test eltype(normalized_laplacian(adj, Float32)) == Float32
    @test neighbors(adj) == [[2,4], [1,3], [2,4], [1,3]]

    adj = [0 2 0 3;
           0 0 4 0;
           2 0 0 1;
           0 0 0 0]
    deg_out = [2 0 0 0;
               0 2 0 0;
               0 0 4 0;
               0 0 0 4]
    deg_in = [5 0 0 0;
              0 4 0 0;
              0 0 3 0;
              0 0 0 0]
    deg_both = [7 0 0 0;
                0 6 0 0;
                0 0 7 0;
                0 0 0 4]

    for T in [Int8, Float64]
        @test degree_matrix(adj, T, dir=:out) == T.(deg_out)
        @test degree_matrix(adj, T, dir=:in) == T.(deg_in)
        @test degree_matrix(adj, T, dir=:both) == T.(deg_both)
        @test_throws DomainError degree_matrix(adj, dir=:other)
        @test laplacian_matrix(adj, T, dir=:out) == T.(deg_out .- adj)
        @test laplacian_matrix(adj, T, dir=:in) == T.(deg_in .- adj)
        @test laplacian_matrix(adj, T, dir=:both) == T.(deg_both .- adj)
    end
    @test neighbors(adj) == [[2,3,4], [1,3], [1,2,4], [1,3]]
end
