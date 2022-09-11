@testset "graphlayers" begin
    @testset "WithGraph" begin
        T = Float32

        adj = T[0. 1. 0. 1.;
                1. 0. 1. 0.;
                0. 1. 0. 1.;
                1. 0. 1. 0.]
        adj2 = T[0. 1. 0. 3.;
                1. 0. 3. 0.;
                0. 3. 0. 1.;
                3. 0. 1. 0.]
        
        fg = FeaturedGraph(adj)
        fg2 = FeaturedGraph(adj2)

        model = Chain(
            GCNConv(32=>32),
            WithGraph(fg2, GCNConv(32=>32)),
            Dense(5, 10),
        )

        model2 = WithGraph(fg, model)

        @test model2[1].graph == GraphSignals.normalized_adjacency_matrix(fg, T; selfloop=true)
        @test model2[2].graph == GraphSignals.normalized_adjacency_matrix(fg2, T; selfloop=true)
        @test model2[3] isa Dense
    end

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
