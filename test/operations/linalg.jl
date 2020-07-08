@testset "linalg" begin
    @testset "symmetric" begin
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
        scaled_lap =   [0 -0.5 0 -0.5;
                        -0.5 0 -0.5 -0;
                        0 -0.5 0 -0.5;
                        -0.5 0 -0.5 0]

        for T in [Int8, Int16, Int32, Int64, Float16, Float32, Float64]
            @test degree_matrix(adj, T, dir=:out) == T.(deg)
            @test degree_matrix(adj, T, dir=:out) == degree_matrix(adj, T, dir=:in)
            @test degree_matrix(adj, T, dir=:out) == degree_matrix(adj, T, dir=:both)
            @test eltype(degree_matrix(adj, T, dir=:out)) == T

            @test laplacian_matrix(adj, T) == T.(lap)
            @test eltype(laplacian_matrix(adj, T)) == T
        end
        for T in [Float16, Float32, Float64]
            @test normalized_laplacian(adj, T) ≈ T.(norm_lap)
            @test eltype(normalized_laplacian(adj, T)) == T
            
            @test scaled_laplacian(adj, T) ≈ T.(scaled_lap)
            @test eltype(scaled_laplacian(adj, T)) == T
        end
        @test neighbors(adj) == [[2,4], [1,3], [2,4], [1,3]]
    end

    @testset "asymmetric" begin
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

        for T in [Int8, Int16, Int32, Int64, Float16, Float32, Float64]
            @test degree_matrix(adj, T, dir=:out) == T.(deg_out)
            @test degree_matrix(adj, T, dir=:in) == T.(deg_in)
            @test degree_matrix(adj, T, dir=:both) == T.(deg_both)
            @test eltype(degree_matrix(adj, T, dir=:out)) == T
            @test eltype(degree_matrix(adj, T, dir=:in)) == T
            @test eltype(degree_matrix(adj, T, dir=:both)) == T
            @test_throws DomainError degree_matrix(adj, dir=:other)

            @test laplacian_matrix(adj, T, dir=:out) == T.(deg_out .- adj)
            @test laplacian_matrix(adj, T, dir=:in) == T.(deg_in .- adj)
            @test laplacian_matrix(adj, T, dir=:both) == T.(deg_both .- adj)
            @test eltype(laplacian_matrix(adj, T, dir=:out)) == T
            @test eltype(laplacian_matrix(adj, T, dir=:in)) == T
            @test eltype(laplacian_matrix(adj, T, dir=:both)) == T
        end
        @test neighbors(adj) == [[2,3,4], [1,3], [1,2,4], [1,3]]
    end
end
