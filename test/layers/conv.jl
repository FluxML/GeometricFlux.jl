@testset "layer" begin
    T = Float32
    in_channel = 3
    in_channel_edge = 1
    out_channel = 5

    N = 4
    E = 4
    adj = T[0. 1. 0. 1.;
            1. 0. 1. 0.;
            0. 1. 0. 1.;
            1. 0. 1. 0.]
    fg = FeaturedGraph(adj)

    @testset "GCNConv" begin
        X = rand(T, in_channel, N)
        Xt = transpose(rand(T, N, in_channel))
        @testset "layer with graph" begin
            gc = GCNConv(fg, in_channel=>out_channel)
            @test size(gc.weight) == (out_channel, in_channel)
            @test size(gc.bias) == (out_channel,)
            @test GraphSignals.adjacency_matrix(gc.fg) == adj

            Y = gc(X)
            @test size(Y) == (out_channel, N)

            # Test with transposed features
            Y = gc(Xt)
            @test size(Y) == (out_channel, N)

            g = Zygote.gradient(() -> sum(gc(X)), Flux.params(gc))
            @test length(g.grads) == 2
        end

        @testset "layer without graph" begin
            gc = GCNConv(in_channel=>out_channel)
            @test size(gc.weight) == (out_channel, in_channel)
            @test size(gc.bias) == (out_channel,)
            @test !has_graph(gc.fg)

            fg = FeaturedGraph(adj, nf=X)
            fg_ = gc(fg)
            @test size(node_feature(fg_)) == (out_channel, N)
            @test_throws ArgumentError gc(X)
            
            # Test with transposed features
            fgt = FeaturedGraph(adj, nf=Xt)
            fgt_ = gc(fgt)
            @test size(node_feature(fgt_)) == (out_channel, N)

            g = Zygote.gradient(() -> sum(node_feature(gc(fg))), Flux.params(gc))
            @test length(g.grads) == 4
        end

        @testset "bias=false" begin
            @test length(Flux.params(GCNConv(2=>3))) == 2
            @test length(Flux.params(GCNConv(2=>3, bias=false))) == 1
        end
    end


    @testset "ChebConv" begin
        k = 6
        X = rand(T, in_channel, N)
        Xt = transpose(rand(T, N, in_channel))
        @testset "layer with graph" begin
            cc = ChebConv(fg, in_channel=>out_channel, k)
            @test size(cc.weight) == (out_channel, in_channel, k)
            @test size(cc.bias) == (out_channel,)
            @test GraphSignals.adjacency_matrix(cc.fg) == adj
            @test cc.k == k
            
            Y = cc(X)
            @test size(Y) == (out_channel, N)

            # Test with transposed features
            Y = cc(Xt)
            @test size(Y) == (out_channel, N)

            g = Zygote.gradient(() -> sum(cc(X)), Flux.params(cc))
            @test length(g.grads) == 2
        end

        @testset "layer without graph" begin
            cc = ChebConv(in_channel=>out_channel, k)
            @test size(cc.weight) == (out_channel, in_channel, k)
            @test size(cc.bias) == (out_channel,)
            @test !has_graph(cc.fg)
            @test cc.k == k
            
            fg = FeaturedGraph(adj, nf=X)
            fg_ = cc(fg)
            @test size(node_feature(fg_)) == (out_channel, N)
            @test_throws ArgumentError cc(X)

            # Test with transposed features
            fgt = FeaturedGraph(adj, nf=Xt)
            fgt_ = cc(fgt)
            @test size(node_feature(fgt_)) == (out_channel, N)

            g = Zygote.gradient(() -> sum(node_feature(cc(fg))), Flux.params(cc))
            @test length(g.grads) == 4
        end

        @testset "bias=false" begin
            @test length(Flux.params(ChebConv(2=>3, 3))) == 2
            @test length(Flux.params(ChebConv(2=>3, 3, bias=false))) == 1
        end
    end

    @testset "GraphConv" begin
        X = rand(T, in_channel, N)
        Xt = transpose(rand(T, N, in_channel))
        @testset "layer with graph" begin
            gc = GraphConv(fg, in_channel=>out_channel)
            @test adjacency_list(gc.fg) == [[2,4], [1,3], [2,4], [1,3]]
            @test size(gc.weight1) == (out_channel, in_channel)
            @test size(gc.weight2) == (out_channel, in_channel)
            @test size(gc.bias) == (out_channel,)

            Y = gc(X)
            @test size(Y) == (out_channel, N)

            # Test with transposed features
            Y = gc(Xt)
            @test size(Y) == (out_channel, N)

            g = Zygote.gradient(() -> sum(gc(X)), Flux.params(gc))
            @test length(g.grads) == 3
        end

        @testset "layer without graph" begin
            gc = GraphConv(in_channel=>out_channel)
            @test size(gc.weight1) == (out_channel, in_channel)
            @test size(gc.weight2) == (out_channel, in_channel)
            @test size(gc.bias) == (out_channel,)

            fg = FeaturedGraph(adj, nf=X)
            fg_ = gc(fg)
            @test size(node_feature(fg_)) == (out_channel, N)
            @test_throws ArgumentError gc(X)

            # Test with transposed features
            fgt = FeaturedGraph(adj, nf=Xt)
            fgt_ = gc(fgt)
            @test size(node_feature(fgt_)) == (out_channel, N)

            g = Zygote.gradient(() -> sum(node_feature(gc(fg))), Flux.params(gc))
            @test length(g.grads) == 5
        end


        @testset "bias=false" begin
            @test length(Flux.params(GraphConv(2=>3))) == 3
            @test length(Flux.params(GraphConv(2=>3, bias=false))) == 2
        end
    end

    @testset "GATConv" begin
        adj1 = [1 1 0 1;
                1 1 1 0;
                0 1 1 1;
                1 0 1 1]
        fg1 = FeaturedGraph(adj1)

        # isolated_vertex
        adj2 = [1 0 0 1;
                0 1 0 0;
                0 0 1 1;
                1 0 1 1]
        fg2 = FeaturedGraph(adj2)

        X = rand(T, in_channel, N)
        Xt = transpose(rand(T, N, in_channel))

        @testset "layer with graph" begin
            for heads = [1, 2], concat = [true, false], adj_gat in [adj1, adj2]
                fg_gat = FeaturedGraph(adj_gat)
                gat = GATConv(fg_gat, in_channel=>out_channel, heads=heads, concat=concat)

                @test size(gat.weight) == (out_channel * heads, in_channel)
                @test size(gat.bias) == (out_channel * heads,)
                @test size(gat.a) == (2*out_channel, heads)

                Y = gat(X)
                @test size(Y) == (concat ? (out_channel*heads, N) : (out_channel, N))

                # Test with transposed features
                Y = gat(Xt)
                @test size(Y) == (concat ? (out_channel*heads, N) : (out_channel, N))

                g = Zygote.gradient(() -> sum(gat(X)), Flux.params(gat))
                @test length(g.grads) == 3
            end
        end

        @testset "layer without graph" begin
            for heads = [1, 2], concat = [true, false], adj_gat in [adj1, adj2]
                fg_gat = FeaturedGraph(adj_gat, nf=X)
                gat = GATConv(in_channel=>out_channel, heads=heads, concat=concat)
                @test size(gat.weight) == (out_channel * heads, in_channel)
                @test size(gat.bias) == (out_channel * heads,)
                @test size(gat.a) == (2*out_channel, heads)

                fg_ = gat(fg_gat)
                Y = node_feature(fg_)
                @test size(Y) == (concat ? (out_channel*heads, N) : (out_channel, N))
                @test_throws ArgumentError gat(X)

                # Test with transposed features
                fgt = FeaturedGraph(adj_gat, nf=Xt)
                fgt_ = gat(fgt)
                @test size(node_feature(fgt_)) == (concat ? (out_channel*heads, N) : (out_channel, N))

                g = Zygote.gradient(() -> sum(node_feature(gat(fg_gat))), Flux.params(gat))
                @test length(g.grads) == 5
            end
        end

        @testset "bias=false" begin
            @test length(Flux.params(GATConv(2=>3))) == 3
            @test length(Flux.params(GATConv(2=>3, bias=false))) == 2
        end
    end

    @testset "GATv2Conv" begin
        adj1 = [1 1 0 1;
                1 1 1 0;
                0 1 1 1;
                1 0 1 1]
        fg1 = FeaturedGraph(adj1)

        # isolated_vertex
        adj2 = [1 0 0 1;
                0 1 0 0;
                0 0 1 1;
                1 0 1 1]
        fg2 = FeaturedGraph(adj2)

        X = rand(T, in_channel, N)
        Xt = transpose(rand(T, N, in_channel))

        @testset "layer with graph" begin
            for heads = [1, 2], concat = [true, false], adj_gat in [adj1, adj2]
                fg_gat = FeaturedGraph(adj_gat)
                gat2 = GATv2Conv(fg_gat, in_channel=>out_channel, heads=heads, concat=concat)

                @test size(gat2.wi) == (out_channel * heads, in_channel)
                @test size(gat2.wi) == (out_channel * heads, in_channel)
                @test size(gat2.biasi) == (out_channel * heads,)
                @test size(gat2.biasj) == (out_channel * heads,)
                @test size(gat2.a) == (out_channel, heads)

                Y = gat2(X)
                @test size(Y) == (concat ? (out_channel*heads, N) : (out_channel, N))

                # Test with transposed features
                Y = gat2(Xt)
                @test size(Y) == (concat ? (out_channel*heads, N) : (out_channel, N))

                g = Zygote.gradient(() -> sum(gat2(X)), Flux.params(gat2))
                @test length(g.grads) == 5
            end
        end

        @testset "layer without graph" begin
            for heads = [1, 2], concat = [true, false], adj_gat in [adj1, adj2]
                fg_gat = FeaturedGraph(adj_gat, nf=X)
                gat2 = GATv2Conv(in_channel=>out_channel, heads=heads, concat=concat)
                @test size(gat2.wi) == (out_channel * heads, in_channel)
                @test size(gat2.wi) == (out_channel * heads, in_channel)
                @test size(gat2.biasi) == (out_channel * heads,)
                @test size(gat2.biasj) == (out_channel * heads,)
                @test size(gat2.a) == (out_channel, heads)

                fg_ = gat2(fg_gat)
                Y = node_feature(fg_)
                @test size(Y) == (concat ? (out_channel*heads, N) : (out_channel, N))
                @test_throws ArgumentError gat2(X)

                # Test with transposed features
                fgt = FeaturedGraph(adj_gat, nf=Xt)
                fgt_ = gat2(fgt)
                @test size(node_feature(fgt_)) == (concat ? (out_channel*heads, N) : (out_channel, N))

                g = Zygote.gradient(() -> sum(node_feature(gat2(fg_gat))), Flux.params(gat2))
                @test length(g.grads) == 7
            end
        end

        @testset "bias=false" begin
            @test length(Flux.params(GATv2Conv(2=>3))) == 5
            @test length(Flux.params(GATv2Conv(2=>3, bias=false))) == 3
        end
    end

    @testset "GatedGraphConv" begin
        num_layers = 3
        X = rand(T, in_channel, N)
        Xt = transpose(rand(T, N, in_channel))
        @testset "layer with graph" begin
            ggc = GatedGraphConv(fg, out_channel, num_layers)
            @test adjacency_list(ggc.fg) == [[2,4], [1,3], [2,4], [1,3]]
            @test size(ggc.weight) == (out_channel, out_channel, num_layers)

            Y = ggc(X)
            @test size(Y) == (out_channel, N)


            # Test with transposed features
            Y = ggc(Xt)
            @test size(Y) == (out_channel, N)

            g = Zygote.gradient(() -> sum(ggc(X)), Flux.params(ggc))
            @test length(g.grads) == 6
        end

        @testset "layer without graph" begin
            ggc = GatedGraphConv(out_channel, num_layers)
            @test size(ggc.weight) == (out_channel, out_channel, num_layers)

            fg = FeaturedGraph(adj, nf=X)
            fg_ = ggc(fg)
            @test size(node_feature(fg_)) == (out_channel, N)
            @test_throws ArgumentError ggc(X)

            # Test with transposed features
            fgt = FeaturedGraph(adj, nf=Xt)
            fgt_ = ggc(fgt)
            @test size(node_feature(fgt_)) == (out_channel, N)

            g = Zygote.gradient(() -> sum(node_feature(ggc(fg))), Flux.params(ggc))
            @test length(g.grads) == 8
        end
    end

    @testset "EdgeConv" begin
        X = rand(T, in_channel, N)
        Xt = transpose(rand(T, N, in_channel))
        @testset "layer with graph" begin
            ec = EdgeConv(fg, Dense(2*in_channel, out_channel))
            @test adjacency_list(ec.fg) == [[2,4], [1,3], [2,4], [1,3]]

            Y = ec(X)
            @test size(Y) == (out_channel, N)

            # Test with transposed features
            Y = ec(Xt)
            @test size(Y) == (out_channel, N)

            g = Zygote.gradient(() -> sum(ec(X)), Flux.params(ec))
            @test length(g.grads) == 2
        end

        @testset "layer without graph" begin
            ec = EdgeConv(Dense(2*in_channel, out_channel))

            fg = FeaturedGraph(adj, nf=X)
            fg_ = ec(fg)
            @test size(node_feature(fg_)) == (out_channel, N)
            @test_throws ArgumentError ec(X)

            # Test with transposed features
            fgt = FeaturedGraph(adj, nf=Xt)
            fgt_ = ec(fgt)
            @test size(node_feature(fgt_)) == (out_channel, N)

            g = Zygote.gradient(() -> sum(node_feature(ec(fg))), Flux.params(ec))
            @test length(g.grads) == 4
        end
    end

    @testset "GINConv" begin
        X = rand(Float32, in_channel, N)
        Xt = transpose(rand(Float32, N, in_channel))
        nn = Flux.Chain(Dense(in_channel, out_channel))
        eps = 0.001

        @testset "layer with graph" begin
            gc = GINConv(FeaturedGraph(adj), nn, eps)
            @test size(gc.nn.layers[1].weight) == (out_channel, in_channel)
            @test size(gc.nn.layers[1].bias) == (out_channel, )
            @test GraphSignals.adjacency_matrix(gc.fg) == adj

            fg = FeaturedGraph(adj, nf=X)
            Y = gc(fg)
            @test size(node_feature(Y)) == (out_channel, N)

            # Test with transposed features
            Y = gc(FeaturedGraph(adj, nf=Xt))
            @test size(node_feature(Y)) == (out_channel, N)

            g = Zygote.gradient(() -> sum(node_feature(gc(fg))), Flux.params(gc))
            @test length(g.grads) == 8
        end
    end

    @testset "CGConv" begin
        fg = FeaturedGraph(adj)
        X = rand(Float32, in_channel, N)
        E = rand(Float32, in_channel_edge, ne(fg))
        Xt = transpose(rand(Float32, N, in_channel))
        @testset "layer with graph" begin
            cgc = CGConv(FeaturedGraph(adj),
                         (in_channel, in_channel_edge))
            @test size(cgc.Wf) == (in_channel, 2 * in_channel + in_channel_edge)
            @test size(cgc.Ws) == (in_channel, 2 * in_channel + in_channel_edge)
            @test size(cgc.bf) == (in_channel,)
            @test size(cgc.bs) == (in_channel,)

            Y = cgc(X, E)
            @test size(Y) == (in_channel, N)

            Yg = cgc(FeaturedGraph(adj, nf=X, ef=E))
            @test size(node_feature(Yg)) == (in_channel, N)
            @test edge_feature(Yg) == E
        end
    end
end
