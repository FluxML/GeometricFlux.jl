using Flux: Dense, gpu

in_channel = 3
out_channel = 5
N = 4
adj = [0 1 0 1;
       1 0 1 0;
       0 1 0 1;
       1 0 1 0]

fg = FeaturedGraph(adj)

@testset "cuda/conv" begin
    @testset "GCNConv" begin
        gc = GCNConv(fg, in_channel=>out_channel) |> gpu
        @test size(gc.weight) == (out_channel, in_channel)
        @test size(gc.bias) == (out_channel,)
        @test collect(graph(gc.fg)) == adj

        X = rand(in_channel, N) |> gpu
        Y = gc(X)
        @test size(Y) == (out_channel, N)

        g = Zygote.gradient(x -> sum(gc(x)), X)[1]
        @test size(g) == size(X)

        g = Zygote.gradient(model -> sum(model(X)), gc)[1]
        @test size(g.weight) == size(gc.weight)
        @test size(g.bias) == size(gc.bias)
    end


    @testset "ChebConv" begin
        k = 6
        cc = ChebConv(fg, in_channel=>out_channel, k) |> gpu
        @test size(cc.weight) == (out_channel, in_channel, k)
        @test size(cc.bias) == (out_channel,)
        @test collect(graph(cc.fg)) == adj
        @test cc.k == k
        @test cc.in_channel == in_channel
        @test cc.out_channel == out_channel

        X = rand(in_channel, N) |> gpu
        Y = cc(X)
        @test size(Y) == (out_channel, N)

        # g = Zygote.gradient(x -> sum(cc(x)), X)[1]
        # @test size(g) == size(X)

        # g = Zygote.gradient(model -> sum(model(X)), cc)[1]
        # @test size(g.weight) == size(cc.weight)
        # @test size(g.bias) == size(cc.bias)
    end

    @testset "GraphConv" begin
        gc = GraphConv(fg, in_channel=>out_channel) |> gpu
        @test size(gc.weight1) == (out_channel, in_channel)
        @test size(gc.weight2) == (out_channel, in_channel)
        @test size(gc.bias) == (out_channel,)

        X = rand(in_channel, N) |> gpu
        Y = gc(X)
        @test size(Y) == (out_channel, N)

        g = Zygote.gradient(x -> sum(gc(x)), X)[1]
        @test size(g) == size(X)

        g = Zygote.gradient(model -> sum(model(X)), gc)[1]
        @test size(g.weight1) == size(gc.weight1)
        @test size(g.weight2) == size(gc.weight2)
        @test size(g.bias) == size(gc.bias)
    end

    @testset "GATConv" begin
        gat = GATConv(fg, in_channel=>out_channel) |> gpu
        @test size(gat.weight) == (out_channel, in_channel)
        @test size(gat.bias) == (out_channel,)

        X = rand(in_channel, N) |> gpu
        Y = gat(X)
        @test size(Y) == (out_channel, N)

        g = Zygote.gradient(x -> sum(gat(x)), X)[1]
        @test size(g) == size(X)

        g = Zygote.gradient(model -> sum(model(X)), gat)[1]
        @test size(g.weight) == size(gat.weight)
        @test size(g.bias) == size(gat.bias)
        @test size(g.a) == size(gat.a)
    end

    @testset "GatedGraphConv" begin
        num_layers = 3
        ggc = GatedGraphConv(fg, out_channel, num_layers) |> gpu
        @test size(ggc.weight) == (out_channel, out_channel, num_layers)

        X = rand(in_channel, N) |> gpu
        Y = ggc(X)
        @test size(Y) == (out_channel, N)

        g = Zygote.gradient(x -> sum(ggc(x)), X)[1]
        @test size(g) == size(X)

        g = Zygote.gradient(model -> sum(model(X)), ggc)[1]
        @test size(g.weight) == size(ggc.weight)
    end

    @testset "EdgeConv" begin
        ec = EdgeConv(fg, Dense(2*in_channel, out_channel)) |> gpu
        X = rand(in_channel, N) |> gpu
        Y = ec(X)
        @test size(Y) == (out_channel, N)

        g = Zygote.gradient(x -> sum(ec(x)), X)[1]
        @test size(g) == size(X)

        g = Zygote.gradient(model -> sum(model(X)), ec)[1]
        @test size(g.nn.weight) == size(ec.nn.weight)
        @test size(g.nn.bias) == size(ec.nn.bias)
    end
end
