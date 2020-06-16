using Flux: Dense

in_channel = 3
out_channel = 5
N = 4
adj = [0. 1. 0. 1.;
       1. 0. 1. 0.;
       0. 1. 0. 1.;
       1. 0. 1. 0.]


@testset "layer" begin
    @testset "GCNConv" begin
        gc = GCNConv(adj, in_channel=>out_channel)
        @test size(gc.weight) == (out_channel, in_channel)
        @test size(gc.bias) == (out_channel,)
        @test graph(gc.graph) === adj

        X = rand(in_channel, N)
        Y = gc(X)
        @test size(Y) == (out_channel, N)

        gc = GCNConv(in_channel=>out_channel)
        @test size(gc.weight) == (out_channel, in_channel)
        @test size(gc.bias) == (out_channel,)
        @test isnothing(graph(gc.graph))

        fg = FeaturedGraph(adj, X)
        fg_ = gc(fg)
        @test size(feature(fg_)) == (out_channel, N)
        @test_throws AssertionError gc(X)
    end


    @testset "ChebConv" begin
        k = 6
        cc = ChebConv(adj, in_channel=>out_channel, k)
        @test size(cc.weight) == (out_channel, in_channel, k)
        @test size(cc.bias) == (out_channel,)
        @test size(cc.L̃) == (N, N)
        @test cc.k == k
        @test cc.in_channel == in_channel
        @test cc.out_channel == out_channel

        X = rand(in_channel, N)
        Y = cc(X)
        @test size(Y) == (out_channel, N)

        # With variable graph
        cc = ChebConv(in_channel=>out_channel, k)
        @test size(cc.weight) == (out_channel, in_channel, k)
        @test size(cc.bias) == (out_channel,)
        @test isnothing(cc.L̃)
        @test cc.k == k
        @test cc.in_channel == in_channel
        @test cc.out_channel == out_channel

        fg = FeaturedGraph(adj, X)
        fg_ = cc(fg)
        @test size(feature(fg_)) == (out_channel, N)
        @test_throws AssertionError cc(X)
    end

    @testset "GraphConv" begin
        gc = GraphConv(adj, in_channel=>out_channel)
        @test gc.adjlist == [[2,4], [1,3], [2,4], [1,3]]
        @test size(gc.weight1) == (out_channel, in_channel)
        @test size(gc.weight2) == (out_channel, in_channel)
        @test size(gc.bias) == (out_channel,)

        X = rand(in_channel, N)
        Y = gc(X)
        @test size(Y) == (out_channel, N)

        # With variable graph
        gc = GraphConv(in_channel=>out_channel)
        @test size(gc.weight1) == (out_channel, in_channel)
        @test size(gc.weight2) == (out_channel, in_channel)
        @test size(gc.bias) == (out_channel,)


        X = rand(in_channel, N)
        fg = FeaturedGraph(adj, X)
        fg_ = gc(fg)
        @test size(feature(fg_)) == (out_channel, N)
        @test_throws MethodError gc(X)
    end

    @testset "GATConv" begin
        for heads = [1, 6]
            for concat = [true, false]
                gat = GATConv(adj, in_channel=>out_channel, heads=heads, concat=concat)
                @test gat.adjlist == [[2,4], [1,3], [2,4], [1,3]]
                @test size(gat.weight) == (out_channel * heads, in_channel)
                @test size(gat.bias) == (out_channel * heads,)
                @test size(gat.a) == (2*out_channel, heads, 1)

                X = rand(in_channel, N)
                Y = gat(X)
                if concat
                    @test size(Y) == (out_channel * heads, N)
                else
                    @test size(Y) == (out_channel * heads, 1)
                end

                # With variable graph
                gat = GATConv(in_channel=>out_channel, heads=heads, concat=concat)
                @test size(gat.weight) == (out_channel * heads, in_channel)
                @test size(gat.bias) == (out_channel * heads,)
                @test size(gat.a) == (2*out_channel, heads, 1)

                X = rand(in_channel, N)
                fg = FeaturedGraph(adj, X)
                fg_ = gat(fg)
                Y = feature(fg_)
                if concat
                    @test size(Y) == (out_channel * heads, N)
                else
                    @test size(Y) == (out_channel * heads, 1)
                end
                @test_throws MethodError gat(X)
            end
        end
    end

    @testset "GatedGraphConv" begin
        num_layers = 3
        ggc = GatedGraphConv(adj, out_channel, num_layers)
        @test ggc.adjlist == [[2,4], [1,3], [2,4], [1,3]]
        @test size(ggc.weight) == (out_channel, out_channel, num_layers)

        X = rand(in_channel, N)
        Y = ggc(X)
        @test size(Y) == (out_channel, N)
    end

    @testset "EdgeConv" begin
        ec = EdgeConv(adj, Dense(2*in_channel, out_channel))
        @test ec.adjlist == [[2,4], [1,3], [2,4], [1,3]]

        X = rand(in_channel, N)
        Y = ec(X)
        @test size(Y) == (out_channel, N)
    end
end
