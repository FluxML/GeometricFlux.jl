@testset "utils" begin
    @testset "GraphParallel" begin
        T = Float32
        N = 4
        E = 5
        adj = T[0 1 1 1;
                1 0 1 0;
                1 1 0 1;
                1 0 1 0]

        nf = rand(3, N)
        ef = rand(5, E)
        gf = rand(7)

        fg = FeaturedGraph(adj, nf=nf, ef=ef, gf=gf)

        layer = GraphParallel(
            node_layer=Dropout(0.5),
            global_layer=x -> x .+ 3.
        )
        fg_ = layer(fg)
        @test node_feature(fg_) == nf
        @test edge_feature(fg_) == ef
        @test global_feature(fg_) == gf .+ 3.
    end
end
