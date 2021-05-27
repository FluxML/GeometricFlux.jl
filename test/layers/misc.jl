num_node = 5
A = rand([0, 1], 5, 5)
A = A .| A'
num_edge = sum(A)
nf = rand(6, num_node)
ef = rand(7, num_edge)
gf = rand(8)

fg = FeaturedGraph(A, nf=nf, ef=ef, gf=gf)

@testset "misc" begin
    @testset "selector" begin
        fs = FeatureSelector(:node)
        @test fs(fg) == nf

        fs = FeatureSelector(:edge)
        @test fs(fg) == ef

        fs = FeatureSelector(:global)
        @test fs(fg) == gf

        @test_throws ArgumentError FeatureSelector(:foo)
    end

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
