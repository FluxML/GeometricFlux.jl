@testset "cuda/conv" begin
    T = Float32
    in_channel = 3
    out_channel = 5
    
    N = 4
    adj = T[0 1 0 1;
           1 0 1 0;
           0 1 0 1;
           1 0 1 0]

    fg = FeaturedGraph(adj)

    @testset "GCNConv" begin
        X = rand(T, in_channel, N)

        @testset "layer without graph" begin
            gc = GCNConv(in_channel=>out_channel) |> gpu
            @test size(gc.weight) == (out_channel, in_channel)
            @test size(gc.bias) == (out_channel,)

            fg = FeaturedGraph(adj, nf=X) |> gpu
            fg_ = gc(fg)
            @test size(node_feature(fg_)) == (out_channel, N)

            g = Zygote.gradient(() -> sum(node_feature(gc(fg))), Flux.params(gc))
            @test length(g.grads) == 4
        end

        @testset "layer with static graph" begin
            gc = WithGraph(fg, GCNConv(in_channel=>out_channel)) |> gpu
            Y = gc(X |> gpu)
            @test size(Y) == (out_channel, N)

            g = Zygote.gradient(() -> sum(gc(X |> gpu)), Flux.params(gc))
            @test length(g.grads) == 3
        end
    end


    @testset "ChebConv" begin
        k = 6
        X = rand(T, in_channel, N)

        @testset "layer without graph" begin
            cc = ChebConv(in_channel=>out_channel, k) |> gpu
            @test size(cc.weight) == (out_channel, in_channel, k)
            @test size(cc.bias) == (out_channel,)
            @test cc.k == k
            
            fg = FeaturedGraph(adj, nf=X) |> gpu
            fg_ = cc(fg)
            @test size(node_feature(fg_)) == (out_channel, N)

            g = Zygote.gradient(() -> sum(node_feature(cc(fg))), Flux.params(cc))
            @test length(g.grads) == 4
        end

        @testset "layer with static graph" begin
            cc = WithGraph(fg, ChebConv(in_channel=>out_channel, k)) |> gpu
            Y = cc(X |> gpu)
            @test size(Y) == (out_channel, N)

            g = Zygote.gradient(() -> sum(cc(X |> gpu)), Flux.params(cc))
            @test length(g.grads) == 3
        end
    end

    @testset "GraphConv" begin
        @testset "layer without graph" begin
            gc = GraphConv(in_channel=>out_channel) |> gpu
            @test size(gc.weight1) == (out_channel, in_channel)
            @test size(gc.weight2) == (out_channel, in_channel)
            @test size(gc.bias) == (out_channel,)

            X = rand(T, in_channel, N)
            fg = FeaturedGraph(adj, nf=X) |> gpu
            fg_ = gc(fg)
            @test size(node_feature(fg_)) == (out_channel, N)

            g = Zygote.gradient(() -> sum(node_feature(gc(fg))), Flux.params(gc))
            @test length(g.grads) == 5
        end

        @testset "layer with static graph" begin
            batch_size = 10
            X = rand(T, in_channel, N, batch_size)
            gc = WithGraph(fg, GraphConv(in_channel=>out_channel)) |> gpu
            Y = gc(X |> gpu)
            @test size(Y) == (out_channel, N, batch_size)

            g = Zygote.gradient(() -> sum(gc(X |> gpu)), Flux.params(gc))
            @test length(g.grads) == 4
        end
    end

    # @testset "GATConv" begin
    #     adj = T[1 1 0 1;
    #             1 1 1 0;
    #             0 1 1 1;
    #             1 0 1 1]
        
    #     fg = FeaturedGraph(adj)

    #     gat = GATConv(fg, in_channel=>out_channel) |> gpu
    #     @test size(gat.weight) == (out_channel, in_channel)
    #     @test size(gat.bias) == (out_channel,)

    #     X = rand(in_channel, N) |> gpu
    #     Y = gat(X)
    #     @test size(Y) == (out_channel, N)

    #     g = Zygote.gradient(() -> sum(gat(X)), Flux.params(gat))
    #     @test length(g.grads) == 3
    # end

    # @testset "GatedGraphConv" begin
    #     num_layers = 3
    #     ggc = GatedGraphConv(fg, out_channel, num_layers) |> gpu
    #     @test size(ggc.weight) == (out_channel, out_channel, num_layers)

    #     X = rand(in_channel, N) |> gpu
    #     Y = ggc(X)
    #     @test size(Y) == (out_channel, N)

    #     g = Zygote.gradient(() -> sum(ggc(X)), Flux.params(ggc))
    #     @test length(g.grads) == 6
    # end

    # @testset "EdgeConv" begin
    #     ec = EdgeConv(fg, Dense(2*in_channel, out_channel)) |> gpu
    #     X = rand(in_channel, N) |> gpu
    #     Y = ec(X)
    #     @test size(Y) == (out_channel, N)

    #     g = Zygote.gradient(() -> sum(ec(X)), Flux.params(ec))
    #     @test length(g.grads) == 2
    # end
end
