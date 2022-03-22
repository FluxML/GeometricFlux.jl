@testset "layer" begin
    T = Float32
    batch_size = 10
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

        @testset "layer without graph" begin
            gc = GCNConv(in_channel=>out_channel)
            @test size(gc.weight) == (out_channel, in_channel)
            @test size(gc.bias) == (out_channel,)

            fg = FeaturedGraph(adj, nf=X)
            fg_ = gc(fg)
            @test size(node_feature(fg_)) == (out_channel, N)
            @test_throws MethodError gc(X)
            
            # Test with transposed features
            fgt = FeaturedGraph(adj, nf=Xt)
            fgt_ = gc(fgt)
            @test size(node_feature(fgt_)) == (out_channel, N)

            g = Zygote.gradient(() -> sum(node_feature(gc(fg))), Flux.params(gc))
            @test length(g.grads) == 4
        end

        @testset "layer with static graph" begin
            gc = WithGraph(fg, GCNConv(in_channel=>out_channel))
            Y = gc(X)
            @test size(Y) == (out_channel, N)

            # Test with transposed features
            Y = gc(Xt)
            @test size(Y) == (out_channel, N)

            g = Zygote.gradient(() -> sum(gc(X)), Flux.params(gc))
            @test length(g.grads) == 3
        end

        @testset "layer with subgraph" begin
            X = rand(T, in_channel, 3)
            nodes = [1,2,4]
            gc = WithGraph(subgraph(fg, nodes), GCNConv(in_channel=>out_channel))
            Y = gc(X)
            @test size(Y) == (out_channel, 3)
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

        @testset "layer without graph" begin
            cc = ChebConv(in_channel=>out_channel, k)
            @test size(cc.weight) == (out_channel, in_channel, k)
            @test size(cc.bias) == (out_channel,)
            @test cc.k == k
            
            fg = FeaturedGraph(adj, nf=X)
            fg_ = cc(fg)
            @test size(node_feature(fg_)) == (out_channel, N)
            @test_throws MethodError cc(X)

            # Test with transposed features
            fgt = FeaturedGraph(adj, nf=Xt)
            fgt_ = cc(fgt)
            @test size(node_feature(fgt_)) == (out_channel, N)

            g = Zygote.gradient(() -> sum(node_feature(cc(fg))), Flux.params(cc))
            @test length(g.grads) == 4
        end

        @testset "layer with static graph" begin
            cc = WithGraph(fg, ChebConv(in_channel=>out_channel, k))            
            Y = cc(X)
            @test size(Y) == (out_channel, N)

            # Test with transposed features
            Y = cc(Xt)
            @test size(Y) == (out_channel, N)

            g = Zygote.gradient(() -> sum(cc(X)), Flux.params(cc))
            @test length(g.grads) == 2
        end

        @testset "bias=false" begin
            @test length(Flux.params(ChebConv(2=>3, 3))) == 2
            @test length(Flux.params(ChebConv(2=>3, 3, bias=false))) == 1
        end
    end

    @testset "GraphConv" begin
        @testset "layer without graph" begin
            gc = GraphConv(in_channel=>out_channel)
            @test size(gc.weight1) == (out_channel, in_channel)
            @test size(gc.weight2) == (out_channel, in_channel)
            @test size(gc.bias) == (out_channel,)

            X = rand(T, in_channel, N)
            fg = FeaturedGraph(adj, nf=X)
            fg_ = gc(fg)
            @test size(node_feature(fg_)) == (out_channel, N)
            @test_throws MethodError gc(X)

            g = Zygote.gradient(() -> sum(node_feature(gc(fg))), Flux.params(gc))
            @test length(g.grads) == 5
        end

        @testset "layer with static graph" begin
            X = rand(T, in_channel, N, batch_size)
            gc = WithGraph(fg, GraphConv(in_channel=>out_channel))
            Y = gc(X)
            @test size(Y) == (out_channel, N, batch_size)

            g = Zygote.gradient(() -> sum(gc(X)), Flux.params(gc))
            @test length(g.grads) == 3
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

        @testset "layer without graph" begin
            for heads = [1, 2], concat = [true, false], adj_gat in [adj1, adj2]
                gat = GATConv(in_channel=>out_channel, heads=heads, concat=concat)
                @test size(gat.weight) == (out_channel * heads, in_channel)
                @test size(gat.bias) == (out_channel * heads,)
                @test size(gat.a) == (2*out_channel, heads)

                X = rand(T, in_channel, N)
                fg_gat = FeaturedGraph(adj_gat, nf=X)
                fg_ = gat(fg_gat)
                @test size(node_feature(fg_)) == (concat ? (out_channel * heads, N) : (out_channel, N))
                @test_throws MethodError gat(X)

                g = Zygote.gradient(() -> sum(node_feature(gat(fg_gat))), Flux.params(gat))
                @test length(g.grads) == 5
            end
        end

        @testset "layer with static graph" begin
            X = rand(T, in_channel, N, batch_size)
            for heads = [1, 2], concat = [true, false], adj_gat in [adj1, adj2]
                fg_gat = FeaturedGraph(adj_gat)
                gat = WithGraph(fg_gat, GATConv(in_channel=>out_channel, heads=heads, concat=concat))
                Y = gat(X)
                @test size(Y) == (concat ? (out_channel * heads, N, batch_size) : (out_channel, N, batch_size))

                g = Zygote.gradient(() -> sum(gat(X)), Flux.params(gat))
                @test length(g.grads) == 3
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

        @testset "layer without graph" begin
            for heads = [1, 2], concat = [true, false], adj_gat in [adj1, adj2]
                gat2 = GATv2Conv(in_channel=>out_channel, heads=heads, concat=concat)
                @test size(gat2.wi) == (out_channel * heads, in_channel)
                @test size(gat2.wi) == (out_channel * heads, in_channel)
                @test size(gat2.biasi) == (out_channel * heads,)
                @test size(gat2.biasj) == (out_channel * heads,)
                @test size(gat2.a) == (out_channel, heads)

                X = rand(T, in_channel, N)
                fg_gat = FeaturedGraph(adj_gat, nf=X)
                fg_ = gat2(fg_gat)
                @test size(node_feature(fg_)) == (concat ? (out_channel * heads, N) : (out_channel, N))
                @test_throws MethodError gat2(X)

                g = Zygote.gradient(() -> sum(node_feature(gat2(fg_gat))), Flux.params(gat2))
                @test length(g.grads) == 7
            end
        end

        @testset "layer with static graph" begin
            X = rand(T, in_channel, N, batch_size)
            for heads = [1, 2], concat = [true, false], adj_gat in [adj1, adj2]
                fg_gat = FeaturedGraph(adj_gat)
                gat2 = WithGraph(fg_gat, GATv2Conv(in_channel=>out_channel, heads=heads, concat=concat))
                Y = gat2(X)
                @test size(Y) == (concat ? (out_channel * heads, N, batch_size) : (out_channel, N, batch_size))

                g = Zygote.gradient(() -> sum(gat2(X)), Flux.params(gat2))
                @test length(g.grads) == 5
            end
        end

        @testset "bias=false" begin
            @test length(Flux.params(GATv2Conv(2=>3))) == 5
            @test length(Flux.params(GATv2Conv(2=>3, bias=false))) == 3
        end
    end

    @testset "GatedGraphConv" begin
        num_layers = 3

        @testset "layer without graph" begin
            ggc = GatedGraphConv(out_channel, num_layers)
            @test size(ggc.weight) == (out_channel, out_channel, num_layers)

            X = rand(T, in_channel, N)
            fg = FeaturedGraph(adj, nf=X)
            fg_ = ggc(fg)
            @test size(node_feature(fg_)) == (out_channel, N)
            @test_throws MethodError ggc(X)

            g = Zygote.gradient(() -> sum(node_feature(ggc(fg))), Flux.params(ggc))
            @test length(g.grads) == 8
        end

        @testset "layer with static graph" begin
            X = rand(T, in_channel, N, batch_size)
            ggc = WithGraph(fg, GatedGraphConv(out_channel, num_layers))
            @test_broken Y = ggc(X)
            @test_broken size(Y) == (out_channel, N, batch_size)

            @test_broken g = Zygote.gradient(() -> sum(ggc(X)), Flux.params(ggc))
            @test_broken length(g.grads) == 6
        end
    end

    @testset "EdgeConv" begin
        @testset "layer without graph" begin
            ec = EdgeConv(Dense(2*in_channel, out_channel))

            X = rand(T, in_channel, N)
            fg = FeaturedGraph(adj, nf=X)
            fg_ = ec(fg)
            @test size(node_feature(fg_)) == (out_channel, N)
            @test_throws MethodError ec(X)

            g = Zygote.gradient(() -> sum(node_feature(ec(fg))), Flux.params(ec))
            @test length(g.grads) == 4
        end
        
        @testset "layer with static graph" begin
            X = rand(T, in_channel, N, batch_size)
            ec = WithGraph(fg, EdgeConv(Dense(2*in_channel, out_channel)))
            Y = ec(X)
            @test size(Y) == (out_channel, N, batch_size)

            g = Zygote.gradient(() -> sum(ec(X)), Flux.params(ec))
            @test length(g.grads) == 2
        end
    end

    @testset "GINConv" begin
        nn = Flux.Chain(Dense(in_channel, out_channel))
        eps = 0.001
        @testset "layer without graph" begin
            gc = GraphConv(in_channel=>out_channel)
            @test size(gc.weight1) == (out_channel, in_channel)
            @test size(gc.weight2) == (out_channel, in_channel)
            @test size(gc.bias) == (out_channel,)

            X = rand(T, in_channel, N)
            fg = FeaturedGraph(adj, nf=X)
            fg_ = gc(fg)
            @test size(node_feature(fg_)) == (out_channel, N)
            @test_throws MethodError gc(X)

            g = Zygote.gradient(() -> sum(node_feature(gc(fg))), Flux.params(gc))
            @test length(g.grads) == 5
        end

        @testset "layer with static graph" begin
            X = rand(T, in_channel, N, batch_size)
            gc = WithGraph(FeaturedGraph(adj), GINConv(nn, eps))
            Y = gc(X)
            @test size(Y) == (out_channel, N, batch_size)

            g = Zygote.gradient(() -> sum(gc(X)), Flux.params(gc))
            @test length(g.grads) == 2
        end
    end

    @testset "CGConv" begin
        @testset "layer without graph" begin
            cgc = CGConv((in_channel, in_channel_edge))
            @test size(cgc.Wf) == (in_channel, 2 * in_channel + in_channel_edge)
            @test size(cgc.Ws) == (in_channel, 2 * in_channel + in_channel_edge)
            @test size(cgc.bf) == (in_channel,)
            @test size(cgc.bs) == (in_channel,)

            nf = rand(T, in_channel, N)
            ef = rand(T, in_channel_edge, E)
            fg = FeaturedGraph(adj, nf=nf, ef=ef)
            fg_ = cgc(fg)
            @test_throws MethodError cgc(nf)

            g = Zygote.gradient(() -> sum(node_feature(cgc(fg))), Flux.params(cgc))
            @test length(g.grads) == 6
        end

        @testset "layer with static graph" begin
            nf = rand(T, in_channel, N, batch_size)
            ef = rand(T, in_channel_edge, E, batch_size)
            cgc = WithGraph(fg, CGConv((in_channel, in_channel_edge)))
            Y = cgc(nf, ef)
            @test size(Y) == (in_channel, N, batch_size)

            g = Zygote.gradient(() -> sum(cgc(nf, ef)), Flux.params(cgc))
            @test length(g.grads) == 4
        end
    end
end
