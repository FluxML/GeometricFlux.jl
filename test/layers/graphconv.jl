@testset "graph conv" begin
    T = Float32
    batch_size = 10
    in_dim = 3
    in_edge_dim = 1
    out_dim = 5

    N = 4
    E = 4
    adj = T[0. 1. 0. 1.;
            1. 0. 1. 0.;
            0. 1. 0. 1.;
            1. 0. 1. 0.]
    fg = FeaturedGraph(adj)

    @testset "GCNConv" begin
        X = rand(T, in_dim, N)
        Xt = transpose(rand(T, N, in_dim))

        @testset "layer without graph" begin
            gc = GCNConv(in_dim=>out_dim)
            @test size(gc.weight) == (out_dim, in_dim)
            @test size(gc.bias) == (out_dim,)

            fg = FeaturedGraph(adj, nf=X)
            fg_ = gc(fg)
            @test size(node_feature(fg_)) == (out_dim, N)
            @test_throws MethodError gc(X)

            # Test with transposed features
            fgt = FeaturedGraph(adj, nf=Xt)
            fgt_ = gc(fgt)
            @test size(node_feature(fgt_)) == (out_dim, N)

            g = gradient(() -> sum(node_feature(gc(fg))), Flux.params(gc))
            @test length(g.grads) == 4
        end

        @testset "layer with static graph" begin
            gc = WithGraph(fg, GCNConv(in_dim=>out_dim))
            Y = gc(X)
            @test size(Y) == (out_dim, N)

            # Test with transposed features
            Y = gc(Xt)
            @test size(Y) == (out_dim, N)

            g = gradient(() -> sum(gc(X)), Flux.params(gc))
            @test length(g.grads) == 3
        end

        @testset "layer with subgraph" begin
            X = rand(T, in_dim, 3)
            nodes = [1,2,4]
            gc = WithGraph(subgraph(fg, nodes), GCNConv(in_dim=>out_dim))
            Y = gc(X)
            @test size(Y) == (out_dim, 3)
        end

        @testset "bias=false" begin
            @test length(Flux.params(GCNConv(2=>3))) == 2
            @test length(Flux.params(GCNConv(2=>3, bias=false))) == 1
        end
    end


    @testset "ChebConv" begin
        k = 6
        X = rand(T, in_dim, N)
        Xt = transpose(rand(T, N, in_dim))

        @testset "layer without graph" begin
            cc = ChebConv(in_dim=>out_dim, k)
            @test size(cc.weight) == (out_dim, in_dim, k)
            @test size(cc.bias) == (out_dim,)
            @test cc.k == k

            fg = FeaturedGraph(adj, nf=X)
            fg_ = cc(fg)
            @test size(node_feature(fg_)) == (out_dim, N)
            @test_throws MethodError cc(X)

            # Test with transposed features
            fgt = FeaturedGraph(adj, nf=Xt)
            fgt_ = cc(fgt)
            @test size(node_feature(fgt_)) == (out_dim, N)

            g = gradient(() -> sum(node_feature(cc(fg))), Flux.params(cc))
            @test length(g.grads) == 4
        end

        @testset "layer with static graph" begin
            cc = WithGraph(fg, ChebConv(in_dim=>out_dim, k))
            Y = cc(X)
            @test size(Y) == (out_dim, N)

            # Test with transposed features
            Y = cc(Xt)
            @test size(Y) == (out_dim, N)

            g = gradient(() -> sum(cc(X)), Flux.params(cc))
            @test length(g.grads) == 2
        end

        @testset "bias=false" begin
            @test length(Flux.params(ChebConv(2=>3, 3))) == 2
            @test length(Flux.params(ChebConv(2=>3, 3, bias=false))) == 1
        end
    end

    @testset "GraphConv" begin
        @testset "layer without graph" begin
            gc = GraphConv(in_dim=>out_dim)
            @test size(gc.weight1) == (out_dim, in_dim)
            @test size(gc.weight2) == (out_dim, in_dim)
            @test size(gc.bias) == (out_dim,)

            X = rand(T, in_dim, N)
            fg = FeaturedGraph(adj, nf=X)
            fg_ = gc(fg)
            @test size(node_feature(fg_)) == (out_dim, N)
            @test_throws MethodError gc(X)

            g = gradient(() -> sum(node_feature(gc(fg))), Flux.params(gc))
            @test length(g.grads) == 5
        end

        @testset "layer with static graph" begin
            X = rand(T, in_dim, N, batch_size)
            gc = WithGraph(fg, GraphConv(in_dim=>out_dim))
            Y = gc(X)
            @test size(Y) == (out_dim, N, batch_size)

            g = gradient(() -> sum(gc(X)), Flux.params(gc))
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

        # isolated_vertex
        adj2 = [1 0 0 1;
                0 1 0 0;
                0 0 1 1;
                1 0 1 1]

        @testset "layer without graph" begin
            for heads = [1, 2], concat = [true, false], adj_gat in [adj1, adj2]
                gat = GATConv(in_dim=>out_dim, heads=heads, concat=concat)
                @test size(gat.weight) == (out_dim * heads, in_dim)
                @test size(gat.bias) == (out_dim * heads,)
                @test size(gat.a) == (2*out_dim, heads)

                X = rand(T, in_dim, N)
                fg_gat = FeaturedGraph(adj_gat, nf=X)
                fg_ = gat(fg_gat)
                @test size(node_feature(fg_)) == (concat ? (out_dim * heads, N) : (out_dim, N))
                @test_throws MethodError gat(X)

                g = gradient(() -> sum(node_feature(gat(fg_gat))), Flux.params(gat))
                @test length(g.grads) == 5
            end
        end

        @testset "layer with static graph" begin
            X = rand(T, in_dim, N, batch_size)
            for heads = [1, 2], concat = [true, false], adj_gat in [adj1, adj2]
                fg_gat = FeaturedGraph(adj_gat)
                gat = WithGraph(fg_gat, GATConv(in_dim=>out_dim, heads=heads, concat=concat))
                Y = gat(X)
                @test size(Y) == (concat ? (out_dim * heads, N, batch_size) : (out_dim, N, batch_size))

                g = gradient(() -> sum(gat(X)), Flux.params(gat))
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

        # isolated_vertex
        adj2 = [1 0 0 1;
                0 1 0 0;
                0 0 1 1;
                1 0 1 1]

        @testset "layer without graph" begin
            for heads = [1, 2], concat = [true, false], adj_gat in [adj1, adj2]
                gat2 = GATv2Conv(in_dim=>out_dim, heads=heads, concat=concat)
                @test size(gat2.wi) == (out_dim * heads, in_dim)
                @test size(gat2.wi) == (out_dim * heads, in_dim)
                @test size(gat2.biasi) == (out_dim * heads,)
                @test size(gat2.biasj) == (out_dim * heads,)
                @test size(gat2.a) == (out_dim, heads)

                X = rand(T, in_dim, N)
                fg_gat = FeaturedGraph(adj_gat, nf=X)
                fg_ = gat2(fg_gat)
                @test size(node_feature(fg_)) == (concat ? (out_dim * heads, N) : (out_dim, N))
                @test_throws MethodError gat2(X)

                g = gradient(() -> sum(node_feature(gat2(fg_gat))), Flux.params(gat2))
                @test length(g.grads) == 7
            end
        end

        @testset "layer with static graph" begin
            X = rand(T, in_dim, N, batch_size)
            for heads = [1, 2], concat = [true, false], adj_gat in [adj1, adj2]
                fg_gat = FeaturedGraph(adj_gat)
                gat2 = WithGraph(fg_gat, GATv2Conv(in_dim=>out_dim, heads=heads, concat=concat))
                Y = gat2(X)
                @test size(Y) == (concat ? (out_dim * heads, N, batch_size) : (out_dim, N, batch_size))

                g = gradient(() -> sum(gat2(X)), Flux.params(gat2))
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
            ggc = GatedGraphConv(out_dim, num_layers)
            @test size(ggc.weight) == (out_dim, out_dim, num_layers)

            X = rand(T, in_dim, N)
            fg = FeaturedGraph(adj, nf=X)
            fg_ = ggc(fg)
            @test size(node_feature(fg_)) == (out_dim, N)
            @test_throws MethodError ggc(X)

            g = gradient(() -> sum(node_feature(ggc(fg))), Flux.params(ggc))
            @test length(g.grads) == 8
        end

        @testset "layer with static graph" begin
            X = rand(T, in_dim, N, batch_size)
            ggc = WithGraph(fg, GatedGraphConv(out_dim, num_layers))
            @test_broken Y = ggc(X)
            @test_broken size(Y) == (out_dim, N, batch_size)

            @test_broken g = gradient(() -> sum(ggc(X)), Flux.params(ggc))
            @test_broken length(g.grads) == 6
        end
    end

    @testset "EdgeConv" begin
        @testset "layer without graph" begin
            ec = EdgeConv(Dense(2*in_dim, out_dim))

            X = rand(T, in_dim, N)
            fg = FeaturedGraph(adj, nf=X)
            fg_ = ec(fg)
            @test size(node_feature(fg_)) == (out_dim, N)
            @test_throws MethodError ec(X)

            g = gradient(() -> sum(node_feature(ec(fg))), Flux.params(ec))
            @test length(g.grads) == 4
        end

        @testset "layer with static graph" begin
            X = rand(T, in_dim, N, batch_size)
            ec = WithGraph(fg, EdgeConv(Dense(2*in_dim, out_dim)))
            Y = ec(X)
            @test size(Y) == (out_dim, N, batch_size)

            g = gradient(() -> sum(ec(X)), Flux.params(ec))
            @test length(g.grads) == 2
        end

        @testset "layer with dynamic graph" begin
            X = rand(T, in_dim, N)
            ec = WithGraph(EdgeConv(Dense(2*in_dim, out_dim)), dynamic=X -> GraphSignals.kneighbors_graph(X, 3))
            Y = ec(X)
            @test size(Y) == (out_dim, N)

            g = gradient(() -> sum(ec(X)), Flux.params(ec))
            @test length(g.grads) == 2
        end

        @testset "layer with dynamic graph in batch" begin
            X = rand(T, in_dim, N, batch_size)
            ec = WithGraph(EdgeConv(Dense(2*in_dim, out_dim)), dynamic=X -> GraphSignals.kneighbors_graph(X, 3))
            Y = ec(X)
            @test size(Y) == (out_dim, N, batch_size)

            g = gradient(() -> sum(ec(X)), Flux.params(ec))
            @test length(g.grads) == 2
        end
    end

    @testset "GINConv" begin
        nn = Flux.Chain(Dense(in_dim, out_dim))
        eps = 0.001
        @testset "layer without graph" begin
            gc = GINConv(nn, eps)
            @test gc.nn == nn

            X = rand(T, in_dim, N)
            fg = FeaturedGraph(adj, nf=X)
            fg_ = gc(fg)
            @test size(node_feature(fg_)) == (out_dim, N)
            @test_throws MethodError gc(X)

            g = gradient(() -> sum(node_feature(gc(fg))), Flux.params(gc))
            @test length(g.grads) == 4
        end

        @testset "layer with static graph" begin
            X = rand(T, in_dim, N, batch_size)
            gc = WithGraph(FeaturedGraph(adj), GINConv(nn, eps))
            Y = gc(X)
            @test size(Y) == (out_dim, N, batch_size)

            g = gradient(() -> sum(gc(X)), Flux.params(gc))
            @test length(g.grads) == 2
        end
    end

    @testset "CGConv" begin
        @testset "layer without graph" begin
            cgc = CGConv((in_dim, in_edge_dim))
            @test size(cgc.f.weight) == (in_dim, 2 * in_dim + in_edge_dim)
            @test size(cgc.s.weight) == (in_dim, 2 * in_dim + in_edge_dim)
            @test size(cgc.f.bias) == (in_dim,)
            @test size(cgc.s.bias) == (in_dim,)

            nf = rand(T, in_dim, N)
            ef = rand(T, in_edge_dim, E)
            fg = FeaturedGraph(adj, nf=nf, ef=ef)
            fg_ = cgc(fg)
            @test_throws MethodError cgc(nf)

            g = gradient(() -> sum(node_feature(cgc(fg))), Flux.params(cgc))
            @test length(g.grads) == 6
        end

        @testset "layer with static graph" begin
            nf = rand(T, in_dim, N, batch_size)
            ef = rand(T, in_edge_dim, E, batch_size)
            cgc = WithGraph(fg, CGConv((in_dim, in_edge_dim)))
            Y = cgc(nf, ef)
            @test size(Y) == (in_dim, N, batch_size)

            g = gradient(() -> sum(cgc(nf, ef)), Flux.params(cgc))
            @test length(g.grads) == 4
        end
    end

    @testset "SAGEConv" begin
        aggregators = [MeanAggregator, MeanPoolAggregator, MaxPoolAggregator,
                       LSTMAggregator]
        @testset "layer without graph" begin
            for conv in aggregators
                l = conv(in_dim=>out_dim, relu, num_sample=3)

                X = rand(T, in_dim, N)
                fg = FeaturedGraph(adj, nf=X)
                fg_ = l(fg)
                @test size(node_feature(fg_)) == (out_dim, N)
                @test_throws MethodError l(X)

                g = gradient(() -> sum(node_feature(l(fg))), Flux.params(l))
                if l.proj == identity
                    if conv == LSTMAggregator
                        @test length(g.grads) == 10
                    else
                        @test length(g.grads) == 5
                    end
                else
                    @test length(g.grads) == 7
                end
            end
        end

        @testset "layer with static graph" begin
            for conv in aggregators
                X = rand(T, in_dim, N, batch_size)
                l = WithGraph(fg, conv(in_dim=>out_dim, relu, num_sample=3))
                if conv == LSTMAggregator
                    @test_throws ArgumentError l(X)
                else
                    Y = l(X)
                    @test size(Y) == (out_dim, N, batch_size)

                    g = gradient(() -> sum(l(X)), Flux.params(l))
                    if l.layer.proj == identity
                        @test length(g.grads) == 3
                    else
                        @test length(g.grads) == 5
                    end
                end
            end
        end
    end

    @testset "GatedGCNConv" begin
        @testset "layer without graph" begin
            l = GatedGCNConv(in_dim=>out_dim)
            @test size(l.A.weight) == (out_dim, in_dim)

            X = rand(T, in_dim, N)
            fg = FeaturedGraph(adj, nf=X)
            fg_ = l(fg)
            @test size(node_feature(fg_)) == (out_dim, N)
            @test_throws MethodError l(X)

            g = gradient(() -> sum(node_feature(l(fg))), Flux.params(l))
            @test length(g.grads) == 6
        end

        @testset "layer with static graph" begin
            X = rand(T, in_dim, N, batch_size)
            l = WithGraph(FeaturedGraph(adj), GatedGCNConv(in_dim=>out_dim))
            Y = l(X)
            @test size(Y) == (out_dim, N, batch_size)

            g = gradient(() -> sum(l(X)), Flux.params(l))
            @test length(g.grads) == 4
        end
    end
end
