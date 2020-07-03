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

@testset "featuredgraphs" begin
    fg = FeaturedGraph(adj)
    @test graph(fg) === adj
    @test length(node_feature(fg)) == 0
    @test nv(fg) == 4

    for T in [Int8, Int16, Int32, Int64, Int128]
        @test degree_matrix(fg, T; dir=:out) == T.(deg)
        @test degree_matrix(fg, T; dir=:out) == degree_matrix(adj, T; dir=:in)
        @test degree_matrix(fg, T; dir=:out) == degree_matrix(adj, T; dir=:both)
        @test laplacian_matrix(fg, T) == T.(lap)
    end
    for T in [Float16, Float32, Float64]
        @test degree_matrix(fg, T; dir=:out) == T.(deg)
        @test degree_matrix(fg, T; dir=:out) == degree_matrix(adj, T; dir=:in)
        @test degree_matrix(fg, T; dir=:out) == degree_matrix(adj, T; dir=:both)
        @test laplacian_matrix(fg, T) == T.(lap)
        @test normalized_laplacian(fg, T) ≈ T.(norm_lap)
        @test scaled_laplacian(fg, T) ≈ T.(scaled_lap)
    end
end
