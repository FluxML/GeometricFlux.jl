@testset "cuda/graph_conv" begin
    T = Float32
    batch_size = 10
    in_dim = 3
    in_edge_dim = 1
    out_dim = 5

    N = 4
    E = 4
    adj = T[0 1 0 1;
           1 0 1 0;
           0 1 0 1;
           1 0 1 0]
    fg = FeaturedGraph(adj)

    @testset "GCNConv" begin
        X = rand(T, in_dim, N)

        @testset "layer without graph" begin
            gc = GCNConv(in_dim=>out_dim) |> gpu
            fg = FeaturedGraph(adj, nf=X) |> gpu
            fg_ = gc(fg)
            @test size(node_feature(fg_)) == (out_dim, N)

            g = gradient(() -> sum(node_feature(gc(fg))), Flux.params(gc))
            @test length(g.grads) == 4
        end

        @testset "layer with static graph" begin
            gc = WithGraph(fg, GCNConv(in_dim=>out_dim)) |> gpu
            Y = gc(X |> gpu)
            @test size(Y) == (out_dim, N)

            g = gradient(() -> sum(gc(X |> gpu)), Flux.params(gc))
            @test length(g.grads) == 3
        end
    end


    @testset "ChebConv" begin
        k = 6
        X = rand(T, in_dim, N)

        @testset "layer without graph" begin
            cc = ChebConv(in_dim=>out_dim, k) |> gpu
            fg = FeaturedGraph(adj, nf=X) |> gpu
            fg_ = cc(fg)
            @test size(node_feature(fg_)) == (out_dim, N)

            g = gradient(() -> sum(node_feature(cc(fg))), Flux.params(cc))
            @test length(g.grads) == 4
        end

        @testset "layer with static graph" begin
            cc = WithGraph(fg, ChebConv(in_dim=>out_dim, k)) |> gpu
            Y = cc(X |> gpu)
            @test size(Y) == (out_dim, N)

            g = gradient(() -> sum(cc(X |> gpu)), Flux.params(cc))
            @test length(g.grads) == 3
        end
    end

    @testset "GraphConv" begin
        @testset "layer without graph" begin
            gc = GraphConv(in_dim=>out_dim) |> gpu
            X = rand(T, in_dim, N)
            fg = FeaturedGraph(adj, nf=X) |> gpu
            fg_ = gc(fg)
            @test size(node_feature(fg_)) == (out_dim, N)

            g = gradient(() -> sum(node_feature(gc(fg))), Flux.params(gc))
            @test length(g.grads) == 5
        end

        @testset "layer with static graph" begin
            X = rand(T, in_dim, N, batch_size)
            gc = WithGraph(fg, GraphConv(in_dim=>out_dim)) |> gpu
            Y = gc(X |> gpu)
            @test size(Y) == (out_dim, N, batch_size)

            g = gradient(() -> sum(gc(X |> gpu)), Flux.params(gc))
            @test length(g.grads) == 4
        end
    end

    @testset "GATConv" begin
        heads = 2
        adj1 = T[1 1 0 1;
                 1 1 1 0;
                 0 1 1 1;
                 1 0 1 1]
        @testset "layer without graph" begin
            gat = GATConv(in_dim=>out_dim, heads=2) |> gpu
            X = rand(T, in_dim, N)
            fg = FeaturedGraph(adj1, nf=X) |> gpu
            fg_ = gat(fg)
            @test size(node_feature(fg_)) == (out_dim * heads, N)

            g = gradient(() -> sum(node_feature(gat(fg))), Flux.params(gat))
            @test length(g.grads) == 5
        end

        @testset "layer with static graph" begin
            X = rand(T, in_dim, N, batch_size)
            fg = FeaturedGraph(adj1)
            gat = WithGraph(fg, GATConv(in_dim=>out_dim, heads=2)) |> gpu
            Y = gat(X |> gpu)
            @test size(Y) == (out_dim * heads, N, batch_size)

            g = gradient(() -> sum(gat(X |> gpu)), Flux.params(gat))
            @test length(g.grads) == 4
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
                l = GATv2Conv(in_dim=>out_dim, heads=heads, concat=concat) |> gpu
                X = rand(T, in_dim, N)
                fg_gat = FeaturedGraph(adj_gat, nf=X) |> gpu
                fg_ = l(fg_gat)
                @test size(node_feature(fg_)) == (concat ? (out_dim * heads, N) : (out_dim, N))
                @test_throws MethodError l(X)

                g = gradient(() -> sum(node_feature(l(fg_gat))), Flux.params(l))
                @test length(g.grads) == 7
            end
        end

        @testset "layer with static graph" begin
            X = rand(T, in_dim, N, batch_size)
            for heads = [1, 2], concat = [true, false], adj_gat in [adj1, adj2]
                fg_gat = FeaturedGraph(adj_gat)
                gat2 = WithGraph(fg_gat, GATv2Conv(in_dim=>out_dim, heads=heads, concat=concat)) |> gpu
                Y = gat2(X |> gpu)
                @test size(Y) == (concat ? (out_dim * heads, N, batch_size) : (out_dim, N, batch_size))

                g = gradient(() -> sum(gat2(X |> gpu)), Flux.params(gat2))
                @test length(g.grads) == 6
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
            ggc = GatedGraphConv(out_dim, num_layers) |> gpu
            X = rand(T, in_dim, N)
            fg = FeaturedGraph(adj, nf=X) |> gpu
            fg_ = ggc(fg)
            @test size(node_feature(fg_)) == (out_dim, N)

            g = gradient(() -> sum(node_feature(ggc(fg))), Flux.params(ggc))
            @test length(g.grads) == 8
        end

        @testset "layer with static graph" begin
            X = rand(T, in_dim, N, batch_size)
            ggc = WithGraph(fg, GatedGraphConv(out_dim, num_layers)) |> gpu
            @test_broken Y = ggc(X |> gpu)
            @test_broken size(Y) == (out_dim, N, batch_size)

            @test_broken g = gradient(() -> sum(ggc(X |> gpu)), Flux.params(ggc))
            @test_broken length(g.grads) == 6
        end
    end

    @testset "EdgeConv" begin
        @testset "layer without graph" begin
            ec = EdgeConv(Dense(2*in_dim, out_dim)) |> gpu
            X = rand(T, in_dim, N)
            fg = FeaturedGraph(adj, nf=X) |> gpu
            fg_ = ec(fg)
            @test size(node_feature(fg_)) == (out_dim, N)

            g = gradient(() -> sum(node_feature(ec(fg))), Flux.params(ec))
            @test length(g.grads) == 4
        end

        @testset "layer with static graph" begin
            X = rand(in_dim, N, batch_size)
            ec = WithGraph(fg, EdgeConv(Dense(2*in_dim, out_dim))) |> gpu
            Y = ec(X |> gpu)
            @test size(Y) == (out_dim, N, batch_size)

            g = gradient(() -> sum(ec(X |> gpu)), Flux.params(ec))
            @test length(g.grads) == 3
        end
    end

    @testset "GINConv" begin
        nn = Flux.Chain(Dense(in_dim, out_dim))
        eps = 0.001
        @testset "layer without graph" begin
            gc = GINConv(nn, eps) |> gpu
            X = rand(T, in_dim, N)
            fg = FeaturedGraph(adj, nf=X) |> gpu
            fg_ = gc(fg)
            @test size(node_feature(fg_)) == (out_dim, N)
            @test_throws MethodError gc(X)

            g = gradient(() -> sum(node_feature(gc(fg))), Flux.params(gc))
            @test length(g.grads) == 4
        end

        @testset "layer with static graph" begin
            X = rand(T, in_dim, N, batch_size)
            gc = WithGraph(FeaturedGraph(adj), GINConv(nn, eps)) |> gpu
            Y = gc(X |> gpu)
            @test size(Y) == (out_dim, N, batch_size)

            g = gradient(() -> sum(gc(X |> gpu)), Flux.params(gc))
            @test length(g.grads) == 3
        end
    end

    @testset "CGConv" begin
        @testset "layer without graph" begin
            cgc = CGConv((in_dim, in_edge_dim)) |> gpu
            nf = rand(T, in_dim, N)
            ef = rand(T, in_edge_dim, E)
            fg = FeaturedGraph(adj, nf=nf, ef=ef) |> gpu
            fg_ = cgc(fg)
            @test_throws MethodError cgc(nf)

            g = gradient(() -> sum(node_feature(cgc(fg))), Flux.params(cgc))
            @test length(g.grads) == 6
        end

        @testset "layer with static graph" begin
            nf = rand(T, in_dim, N, batch_size)
            ef = rand(T, in_edge_dim, E, batch_size)
            cgc = WithGraph(fg, CGConv((in_dim, in_edge_dim))) |> gpu
            Y = cgc(nf |> gpu, ef |> gpu)
            @test size(Y) == (in_dim, N, batch_size)

            g = gradient(() -> sum(cgc(nf |> gpu, ef |> gpu)), Flux.params(cgc))
            @test length(g.grads) == 6
        end
    end

    @testset "SAGEConv" begin
        aggregators = [MeanAggregator, MeanPoolAggregator, MaxPoolAggregator,
                       LSTMAggregator]
        @testset "layer without graph" begin
            X = rand(T, in_dim, N)
            fg = FeaturedGraph(adj, nf=X) |> gpu

            for conv in aggregators
                l = conv(in_dim=>out_dim, relu, num_sample=3) |> gpu
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
            X = rand(T, in_dim, N, batch_size) |> gpu
            for conv in aggregators
                l = WithGraph(fg, conv(in_dim=>out_dim, relu, num_sample=3)) |> gpu
                if conv == LSTMAggregator
                    @test_throws ArgumentError l(X)
                else
                    Y = l(X |> gpu)
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
            l = GatedGCNConv(in_dim=>out_dim) |> gpu
            X = rand(T, in_dim, N)
            fg = FeaturedGraph(adj, nf=X) |> gpu
            fg_ = l(fg)
            @test size(node_feature(fg_)) == (out_dim, N)
            @test_throws MethodError l(X)

            g = gradient(() -> sum(node_feature(l(fg))), Flux.params(l))
            @test length(g.grads) == 6
        end

        @testset "layer with static graph" begin
            X = rand(T, in_dim, N, batch_size)
            l = WithGraph(FeaturedGraph(adj), GatedGCNConv(in_dim=>out_dim)) |> gpu
            Y = l(X |> gpu)
            @test size(Y) == (out_dim, N, batch_size)

            g = gradient(() -> sum(l(X |> gpu)), Flux.params(l))
            @test length(g.grads) == 5
        end
    end
end
