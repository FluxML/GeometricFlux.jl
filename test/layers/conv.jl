using Flux: Dense, Chain
in_channel = 3
out_channel = 5
N = 4
T = Float32
adj = T[0. 1. 0. 1.;
       1. 0. 1. 0.;
       0. 1. 0. 1.;
       1. 0. 1. 0.]

adj_single_vertex =   T[0. 0. 0. 1.;
                        0. 0. 0. 0.;
                        0. 0. 0. 1.;
                        1. 0. 1. 0.]


@testset "layer" begin
    @testset "GCNConv" begin
        X = rand(Float32, in_channel, N)
        Xt = transpose(rand(Float32, N, in_channel))
        @testset "layer with graph" begin
            gc = GCNConv(adj, in_channel=>out_channel)
            @test size(gc.weight) == (out_channel, in_channel)
            @test size(gc.bias) == (out_channel,)
            @test graph(gc.fg) === adj

            Y = gc(X)
            @test size(Y) == (out_channel, N)

            # Test with transposed features
            Y = gc(Xt)
            @test size(Y) == (out_channel, N)

            g = Zygote.gradient(x -> sum(gc(x)), X)[1]
            @test size(g) == size(X)

            g = Zygote.gradient(model -> sum(model(X)), gc)[1]
            @test size(g.weight) == size(gc.weight)
            @test size(g.bias) == size(gc.bias)
        end

        @testset "layer without graph" begin
            gc = GCNConv(in_channel=>out_channel)
            @test size(gc.weight) == (out_channel, in_channel)
            @test size(gc.bias) == (out_channel,)
            @test !has_graph(gc.fg)

            fg = FeaturedGraph(adj, nf=X)
            fg_ = gc(fg)
            @test size(node_feature(fg_)) == (out_channel, N)
            @test_throws AssertionError gc(X)

            # Test with transposed features
            fgt = FeaturedGraph(adj, nf=Xt)
            fgt_ = gc(fgt)
            @test size(node_feature(fgt_)) == (out_channel, N)

            g = Zygote.gradient(x -> sum(node_feature(gc(x))), fg)[1]
            @test size(g[].nf) == size(X)

            g = Zygote.gradient(model -> sum(node_feature(model(fg))), gc)[1]
            @test size(g.weight) == size(gc.weight)
            @test size(g.bias) == size(gc.bias)
        end
    end


    @testset "ChebConv" begin
        k = 6
        X = rand(Float32, in_channel, N)
        Xt = transpose(rand(Float32, N, in_channel))
        @testset "layer with graph" begin
            cc = ChebConv(adj, in_channel=>out_channel, k)
            @test size(cc.weight) == (out_channel, in_channel, k)
            @test size(cc.bias) == (out_channel,)
            @test graph(cc.fg) === adj
            @test cc.k == k
            @test cc.in_channel == in_channel
            @test cc.out_channel == out_channel

            Y = cc(X)
            @test size(Y) == (out_channel, N)

            # Test with transposed features
            Y = cc(Xt)
            @test size(Y) == (out_channel, N)

            g = Zygote.gradient(x -> sum(cc(x)), X)[1]
            @test size(g) == size(X)

            g = Zygote.gradient(model -> sum(model(X)), cc)[1]
            @test size(g.weight) == size(cc.weight)
            @test size(g.bias) == size(cc.bias)
        end

        @testset "layer without graph" begin
            cc = ChebConv(in_channel=>out_channel, k)
            @test size(cc.weight) == (out_channel, in_channel, k)
            @test size(cc.bias) == (out_channel,)
            @test !has_graph(cc.fg)
            @test cc.k == k
            @test cc.in_channel == in_channel
            @test cc.out_channel == out_channel

            fg = FeaturedGraph(adj, nf=X)
            fg_ = cc(fg)
            @test size(node_feature(fg_)) == (out_channel, N)
            @test_throws AssertionError cc(X)

            # Test with transposed features
            fgt = FeaturedGraph(adj, nf=Xt)
            fgt_ = cc(fgt)
            @test size(node_feature(fgt_)) == (out_channel, N)

            g = Zygote.gradient(x -> sum(node_feature(cc(x))), fg)[1]
            @test size(g[].nf) == size(X)

            g = Zygote.gradient(model -> sum(node_feature(model(fg))), cc)[1]
            @test size(g.weight) == size(cc.weight)
            @test size(g.bias) == size(cc.bias)
        end
    end

    @testset "GraphConv" begin
        X = rand(Float32, in_channel, N)
        Xt = transpose(rand(Float32, N, in_channel))
        @testset "layer with graph" begin
            gc = GraphConv(adj, in_channel=>out_channel)
            @test graph(gc.fg) == [[2,4], [1,3], [2,4], [1,3]]
            @test size(gc.weight1) == (out_channel, in_channel)
            @test size(gc.weight2) == (out_channel, in_channel)
            @test size(gc.bias) == (out_channel,)

            Y = gc(X)
            @test size(Y) == (out_channel, N)

            # Test with transposed features
            Y = gc(Xt)
            @test size(Y) == (out_channel, N)

            g = Zygote.gradient(x -> sum(gc(x)), X)[1]
            @test size(g) == size(X)

            g = Zygote.gradient(model -> sum(model(X)), gc)[1]
            @test size(g.weight1) == size(gc.weight1)
            @test size(g.weight2) == size(gc.weight2)
            @test size(g.bias) == size(gc.bias)
        end

        @testset "layer without graph" begin
            gc = GraphConv(in_channel=>out_channel)
            @test size(gc.weight1) == (out_channel, in_channel)
            @test size(gc.weight2) == (out_channel, in_channel)
            @test size(gc.bias) == (out_channel,)

            fg = FeaturedGraph(adj, nf=X)
            fg_ = gc(fg)
            @test size(node_feature(fg_)) == (out_channel, N)
            @test_throws AssertionError gc(X)

            # Test with transposed features
            fgt = FeaturedGraph(adj, nf=Xt)
            fgt_ = gc(fgt)
            @test size(node_feature(fgt_)) == (out_channel, N)

            g = Zygote.gradient(x -> sum(node_feature(gc(x))), fg)[1]
            @test size(g[].nf) == size(X)

            g = Zygote.gradient(model -> sum(node_feature(model(fg))), gc)[1]
            @test size(g.weight1) == size(gc.weight1)
            @test size(g.weight2) == size(gc.weight2)
            @test size(g.bias) == size(gc.bias)
        end
    end

    @testset "GATConv" begin

        X = rand(Float32, in_channel, N)
        Xt = transpose(rand(Float32, N, in_channel))

        for adj_gat in [adj, adj_single_vertex]

            @testset "layer with graph" begin
                @testset "concat=true" begin
                    for heads = [1, 6]
                        gat = GATConv(adj_gat, in_channel=>out_channel, heads=heads, concat=true)

                        if adj_gat == adj
                            @test graph(gat.fg) == [[2,4], [1,3], [2,4], [1,3]]
                        end
                        if adj_gat == adj_single_vertex
                            @test graph(gat.fg) == [[4], Int64[], [4], [1, 3]]
                        end

                        @test size(gat.weight) == (out_channel * heads, in_channel)
                        @test size(gat.bias) == (out_channel * heads,)
                        @test size(gat.a) == (2*out_channel, heads)

                        Y = gat(X)
                        @test size(Y) == (out_channel * heads, N)

                        # Test with transposed features
                        Y = gat(Xt)
                        @test size(Y) == (out_channel * heads, N)

                        g = Zygote.gradient(x -> sum(gat(x)), X)[1]
                        @test size(g) == size(X)

                        g = Zygote.gradient(model -> sum(model(X)), gat)[1]
                        @test size(g.weight) == size(gat.weight)
                        @test size(g.bias) == size(gat.bias)
                        @test size(g.a) == size(gat.a)
                    end
                end

                @testset "concat=false" begin
                    for heads = [1, 6]
                        gat = GATConv(adj_gat, in_channel=>out_channel, heads=heads, concat=false)
                        if adj_gat == adj
                            @test graph(gat.fg) == [[2,4], [1,3], [2,4], [1,3]]
                        end
                        if adj_gat == adj_single_vertex
                            @test graph(gat.fg) == [[4], Int64[], [4], [1, 3]]
                        end
                        @test size(gat.weight) == (out_channel * heads, in_channel)
                        @test size(gat.bias) == (out_channel * heads,)
                        @test size(gat.a) == (2*out_channel, heads)

                        Y = gat(X)
                        @test size(Y) == (out_channel, N)

                        # Test with transposed features
                        Y = gat(Xt)
                        @test size(Y) == (out_channel, N)

                        g = Zygote.gradient(x -> sum(gat(x)), X)[1]
                        @test size(g) == size(X)

                        g = Zygote.gradient(model -> sum(model(X)), gat)[1]
                        @test size(g.weight) == size(gat.weight)
                        @test size(g.bias) == size(gat.bias)
                        @test size(g.a) == size(gat.a)
                    end
                end
            end

            @testset "layer without graph" begin
                fg = FeaturedGraph(adj_gat, nf=X)

                @testset "concat=true" begin
                    for heads = [1, 6]
                        gat = GATConv(in_channel=>out_channel, heads=heads, concat=true)
                        @test size(gat.weight) == (out_channel * heads, in_channel)
                        @test size(gat.bias) == (out_channel * heads,)
                        @test size(gat.a) == (2*out_channel, heads)

                        fg_ = gat(fg)
                        Y = node_feature(fg_)
                        @test size(Y) == (out_channel * heads, N)
                        @test_throws AssertionError gat(X)

                        # Test with transposed features
                        fgt = FeaturedGraph(adj_gat, nf=Xt)
                        fgt_ = gat(fgt)
                        @test size(node_feature(fgt_)) == (out_channel * heads, N)

                        g = Zygote.gradient(x -> sum(node_feature(gat(x))), fg)[1]
                        @test size(g[].nf) == size(X)

                        g = Zygote.gradient(model -> sum(node_feature(model(fg))), gat)[1]
                        @test size(g.weight) == size(gat.weight)
                        @test size(g.bias) == size(gat.bias)
                        @test size(g.a) == size(gat.a)
                    end
                end

                @testset "concat=false" begin
                    for heads = [1, 6]
                        gat = GATConv(in_channel=>out_channel, heads=heads, concat=false)
                        @test size(gat.weight) == (out_channel * heads, in_channel)
                        @test size(gat.bias) == (out_channel * heads,)
                        @test size(gat.a) == (2*out_channel, heads)

                        fg_ = gat(fg)
                        Y = node_feature(fg_)
                        @test size(Y) == (out_channel, N)
                        @test_throws AssertionError gat(X)

                        # Test with transposed features
                        fgt = FeaturedGraph(adj_gat, nf=Xt)
                        fgt_ = gat(fgt)
                        @test size(node_feature(fgt_)) == (out_channel, N)

                        g = Zygote.gradient(x -> sum(node_feature(gat(x))), fg)[1]
                        @test size(g[].nf) == size(X)

                        g = Zygote.gradient(model -> sum(node_feature(model(fg))), gat)[1]
                        @test size(g.weight) == size(gat.weight)
                        @test size(g.bias) == size(gat.bias)
                        @test size(g.a) == size(gat.a)
                    end
                end
            end
        end
    end

    @testset "GatedGraphConv" begin
        num_layers = 3
        X = rand(Float32, in_channel, N)
        Xt = transpose(rand(Float32, N, in_channel))
        @testset "layer with graph" begin
            ggc = GatedGraphConv(adj, out_channel, num_layers)
            @test graph(ggc.fg) == [[2,4], [1,3], [2,4], [1,3]]
            @test size(ggc.weight) == (out_channel, out_channel, num_layers)

            Y = ggc(X)
            @test size(Y) == (out_channel, N)


            # Test with transposed features
            Y = ggc(Xt)
            @test size(Y) == (out_channel, N)

            g = Zygote.gradient(x -> sum(ggc(x)), X)[1]
            @test size(g) == size(X)

            g = Zygote.gradient(model -> sum(model(X)), ggc)[1]
            @test size(g.weight) == size(ggc.weight)
        end

        @testset "layer without graph" begin
            ggc = GatedGraphConv(out_channel, num_layers)
            @test size(ggc.weight) == (out_channel, out_channel, num_layers)

            fg = FeaturedGraph(adj, nf=X)
            fg_ = ggc(fg)
            @test size(node_feature(fg_)) == (out_channel, N)
            @test_throws AssertionError ggc(X)


            # Test with transposed features
            fgt = FeaturedGraph(adj, nf=Xt)
            fgt_ = ggc(fgt)
            @test size(node_feature(fgt_)) == (out_channel, N)

            g = Zygote.gradient(x -> sum(node_feature(ggc(x))), fg)[1]
            @test size(g[].nf) == size(X)

            g = Zygote.gradient(model -> sum(node_feature(model(fg))), ggc)[1]
            @test size(g.weight) == size(ggc.weight)
        end
    end

    @testset "EdgeConv" begin
        X = rand(Float32, in_channel, N)
        Xt = transpose(rand(Float32, N, in_channel))
        @testset "layer with graph" begin
            ec = EdgeConv(adj, Dense(2*in_channel, out_channel))
            @test graph(ec.fg) == [[2,4], [1,3], [2,4], [1,3]]

            Y = ec(X)
            @test size(Y) == (out_channel, N)


            # Test with transposed features
            Y = ec(Xt)
            @test size(Y) == (out_channel, N)

            g = Zygote.gradient(x -> sum(ec(x)), X)[1]
            @test size(g) == size(X)

            g = Zygote.gradient(model -> sum(model(X)), ec)[1]
            @test size(g.nn.W) == size(ec.nn.W)
            @test size(g.nn.b) == size(ec.nn.b)
        end

        @testset "layer without graph" begin
            ec = EdgeConv(Dense(2*in_channel, out_channel))

            fg = FeaturedGraph(adj, nf=X)
            fg_ = ec(fg)
            @test size(node_feature(fg_)) == (out_channel, N)
            @test_throws AssertionError ec(X)


            # Test with transposed features
            fgt = FeaturedGraph(adj, nf=Xt)
            fgt_ = ec(fgt)
            @test size(node_feature(fgt_)) == (out_channel, N)

            g = Zygote.gradient(x -> sum(node_feature(ec(x))), fg)[1]
            @test size(g[].nf) == size(X)

            g = Zygote.gradient(model -> sum(node_feature(model(fg))), ec)[1]
            @test size(g.nn.W) == size(ec.nn.W)
            @test size(g.nn.b) == size(ec.nn.b)
        end
    end

    @testset "GINConv" begin
        X = rand(Float32, in_channel, N)
        Xt = transpose(rand(Float32, N, in_channel))
        nn = Flux.Chain(Dense(in_channel, out_channel))
        eps = 0.001

        @testset "layer with graph" begin
            gc = GINConv(FeaturedGraph(adj), nn, eps)
            @test size(gc.nn.layers[1].W) == (out_channel, in_channel)
            @test size(gc.nn.layers[1].b) == (out_channel, )
            @test graph(gc.fg) === adj

            Y = gc(FeaturedGraph(adj, nf=X))
            @test size(node_feature(Y)) == (out_channel, N)

            # Test with transposed features
            Y = gc(FeaturedGraph(adj, nf=Xt))
            @test size(node_feature(Y)) == (out_channel, N)

            g = Zygote.gradient(x -> sum(node_feature(gc(x))), 
                                FeaturedGraph(adj, nf=X))[1]
            @test size(g.x.nf) == size(X)

            g = Zygote.gradient(model -> sum(node_feature(model(FeaturedGraph(adj, nf=X)))), 
                                gc)[1]
            @test size(g.nn.layers[1].W) == size(gc.nn.layers[1].W)
            @test size(g.nn.layers[1].b) == size(gc.nn.layers[1].b)
            @test g.eps === nothing
        end
    end
end
