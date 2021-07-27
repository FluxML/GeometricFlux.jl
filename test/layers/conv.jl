using Flux: Dense, Chain
in_channel = 3
out_channel = 5
N = 4
T = Float32
adj = T[0. 1. 0. 1.;
       1. 0. 1. 0.;
       0. 1. 0. 1.;
       1. 0. 1. 0.]

fg = FeaturedGraph(adj)
    
adj_single_vertex = T[0. 0. 0. 1.;
                      0. 0. 0. 0.;
                      0. 0. 0. 1.;
                      1. 0. 1. 0.]

fg_single_vertex = FeaturedGraph(adj_single_vertex)
            

@testset "layer" begin
    @testset "GCNConv" begin
        X = rand(T, in_channel, N)
        Xt = transpose(rand(T, N, in_channel))
        @testset "layer with graph" begin
            gc = GCNConv(fg, in_channel=>out_channel)
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
            @test_throws MethodError gc(X)
            
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

        @testset "bias=false" begin
            length(Flux.params(GCNConv(2=>3))) == 2
            length(Flux.params(GCNConv(2=>3, bias=false))) == 1
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
            @test_throws MethodError cc(X)

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

        @testset "bias=false" begin
            length(Flux.params(ChebConv(2=>3, 3))) == 2
            length(Flux.params(ChebConv(2=>3, 3, bias=false))) == 1
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
            @test_throws MethodError gc(X)

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


        @testset "bias=false" begin
            length(Flux.params(GraphConv(2=>3))) == 3
            length(Flux.params(GraphConv(2=>3, bias=false))) == 2
        end
    end

    @testset "GATConv" begin

        X = rand(T, in_channel, N)
        Xt = transpose(rand(T, N, in_channel))

        @testset "layer with graph" begin
            for heads = [1, 2], concat = [true, false], adj_gat in [adj, adj_single_vertex]
                fg_gat = FeaturedGraph(adj_gat)
                gat = GATConv(fg_gat, in_channel=>out_channel, heads=heads, concat=concat)

                if adj_gat == adj
                    @test adjacency_list(gat.fg) == [[2,4], [1,3], [2,4], [1,3]]
                elseif adj_gat == adj_single_vertex
                    @test adjacency_list(gat.fg) == [[4], Int64[], [4], [1, 3]]
                end

                @test size(gat.weight) == (out_channel * heads, in_channel)
                @test size(gat.bias) == (out_channel * heads,)
                @test size(gat.a) == (2*out_channel, heads)

                Y = gat(X)
                @test size(Y) == (concat ? (out_channel*heads, N) : (out_channel, N))

                # Test with transposed features
                Y = gat(Xt)
                @test size(Y) == (concat ? (out_channel*heads, N) : (out_channel, N))

                g = Zygote.gradient(x -> sum(gat(x)), X)[1]
                @test size(g) == size(X)

                g = Zygote.gradient(model -> sum(model(X)), gat)[1]
                @test size(g.weight) == size(gat.weight)
                @test size(g.bias) == size(gat.bias)
                @test size(g.a) == size(gat.a)
            end
        end

        @testset "layer without graph" begin
            for heads = [1, 2], concat = [true, false], adj_gat in [adj, adj_single_vertex]
                fg_gat = FeaturedGraph(adj_gat, nf=X)
                gat = GATConv(in_channel=>out_channel, heads=heads, concat=concat)
                @test size(gat.weight) == (out_channel * heads, in_channel)
                @test size(gat.bias) == (out_channel * heads,)
                @test size(gat.a) == (2*out_channel, heads)

                fg_ = gat(fg_gat)
                Y = node_feature(fg_)
                @test size(Y) == (concat ? (out_channel*heads, N) : (out_channel, N))
                @test_throws MethodError gat(X)

                # Test with transposed features
                fgt = FeaturedGraph(adj_gat, nf=Xt)
                fgt_ = gat(fgt)
                @test size(node_feature(fgt_)) == (concat ? (out_channel*heads, N) : (out_channel, N))

                g = Zygote.gradient(x -> sum(node_feature(gat(x))), fg_gat)[1]
                @test size(g[].nf) == size(X)

                g = Zygote.gradient(model -> sum(node_feature(model(fg_gat))), gat)[1]
                @test size(g.weight) == size(gat.weight)
                @test size(g.bias) == size(gat.bias)
                @test size(g.a) == size(gat.a)
            end
        end

        @testset "bias=false" begin
            length(Flux.params(GATConv(2=>3))) == 3
            length(Flux.params(GATConv(2=>3, bias=false))) == 2
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
            @test_throws MethodError ggc(X)

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

            g = Zygote.gradient(x -> sum(ec(x)), X)[1]
            @test size(g) == size(X)

            g = Zygote.gradient(model -> sum(model(X)), ec)[1]
            @test size(g.nn.weight) == size(ec.nn.weight)
            @test size(g.nn.bias) == size(ec.nn.bias)
        end

        @testset "layer without graph" begin
            ec = EdgeConv(Dense(2*in_channel, out_channel))

            fg = FeaturedGraph(adj, nf=X)
            fg_ = ec(fg)
            @test size(node_feature(fg_)) == (out_channel, N)
            @test_throws MethodError ec(X)

            # Test with transposed features
            fgt = FeaturedGraph(adj, nf=Xt)
            fgt_ = ec(fgt)
            @test size(node_feature(fgt_)) == (out_channel, N)

            g = Zygote.gradient(x -> sum(node_feature(ec(x))), fg)[1]
            @test size(g[].nf) == size(X)

            g = Zygote.gradient(model -> sum(node_feature(model(fg))), ec)[1]
            @test size(g.nn.weight) == size(ec.nn.weight)
            @test size(g.nn.bias) == size(ec.nn.bias)
        end
    end

    @testset "GINConv" begin
        X = rand(Float32, in_channel, N)
        Xt = transpose(rand(Float32, N, in_channel))
        nn = Flux.Chain(Dense(in_channel, out_channel))
        eps = 0.001

        @testset "layer with graph" begin
            gc = GINConv(FeaturedGraph(adj), nn, eps=eps)
            @test size(gc.nn.layers[1].weight) == (out_channel, in_channel)
            @test size(gc.nn.layers[1].bias) == (out_channel, )
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
            @test size(g.nn.layers[1].weight) == size(gc.nn.layers[1].weight)
            @test size(g.nn.layers[1].bias) == size(gc.nn.layers[1].bias)
            @test !in(:eps, Flux.trainable(gc))
        end
    end
end
