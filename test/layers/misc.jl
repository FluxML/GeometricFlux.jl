@testset "misc" begin
    @testset "bypass_graph" begin
        N = 4
        E = 5
        adj = [0 1 1 1;
            1 0 1 0;
            1 1 0 1;
            1 0 1 0]

        nf = rand(3, N)
        ef = rand(5, E)
        gf = rand(7)

        fg = FeaturedGraph(adj, nf=nf, ef=ef, gf=gf)
        layer = bypass_graph(x -> x .+ 1.,
                                x -> x .+ 2.,
                                x -> x .+ 3.)
        fg_ = layer(fg)
        @test graph(fg_) == adj
        @test node_feature(fg_) == nf .+ 1.
        @test edge_feature(fg_) == ef .+ 2.
        @test global_feature(fg_) == gf .+ 3.
    end
end
