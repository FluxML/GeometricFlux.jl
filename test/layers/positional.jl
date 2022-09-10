@testset "positional" begin
    T = Float32
    batch_size = 10
    in_channel = 3
    in_channel_edge = 2
    pos_channel = 3
    out_channel = 5

    N = 4
    E = 4
    adj = T[0. 1. 0. 1.;
            1. 0. 1. 0.;
            0. 1. 0. 1.;
            1. 0. 1. 0.]
    fg = FeaturedGraph(adj)

    @testset "EEquivGraphPE" begin
        @testset "layer without graph" begin
            l = EEquivGraphPE(in_channel_edge=>out_channel)

            nf = rand(T, out_channel, N)
            ef = rand(T, in_channel_edge, E)
            fg = FeaturedGraph(adj, nf=nf, ef=ef)
            fg_ = l(fg)
            @test size(node_feature(fg_)) == (out_channel, N)

            g = gradient(() -> sum(node_feature(l(fg))), Flux.params(l))
            @test length(g.grads) == 4
        end

        @testset "layer with static graph" begin
            nf = rand(T, out_channel, N, batch_size)
            ef = rand(T, in_channel_edge, E, batch_size)
            l = WithGraph(fg, EEquivGraphPE(in_channel_edge=>out_channel))
            Y = l(nf, ef)
            @test size(Y) == (out_channel, N, batch_size)

            g = gradient(() -> sum(l(nf, ef)), Flux.params(l))
            @test length(g.grads) == 2
        end
    end

    @testset "PositionalEncoding" begin
        K = 3
        l = PositionalEncoding(fg, K)
        @test size(l.pe) == (K, N)

        fg_ = l(fg)
        @test GraphSignals.has_positional_feature(fg_)
    end

    @testset "LSPE" begin
        @testset "layer not accepts edge features" begin
            f_h = GatedGCNConv(in_channel + pos_channel=>out_channel, residual=true)
            f_e = Dense(2in_channel, out_channel)
            f_p = Dense(pos_channel, pos_channel)
            l = LSPE(f_h, f_e, f_p)

            nf = rand(T, in_channel, N)
            pf = rand(T, pos_channel, N)
            fg = FeaturedGraph(adj, nf=vcat(nf, pf), pf=pf)
            fg_ = l(fg)
            @test size(node_feature(fg_)) == (out_channel, N)
            @test size(positional_feature(fg_)) == (pos_channel, N)

            g = gradient(() -> sum(positional_feature(l(fg))), Flux.params(l))
            @test length(g.grads) == 2
        end

        # @testset "layer accepts edge features" begin
        #     f_h = GraphConv(in_channel=>out_channel)
        #     f_e = Dense(in_channel_edge, out_channel)
        #     f_p = Dense(pos_channel, pos_channel)
        #     l = LSPE(f_h, f_e, f_p)

        #     nf = rand(T, in_channel, N)
        #     ef = rand(T, in_channel_edge, E)
        #     pf = rand(T, pos_channel, N)
        #     fg = FeaturedGraph(adj, nf=nf, ef=ef, pf=pf)
        #     fg_ = l(fg)
        #     @test size(node_feature(fg_)) == (out_channel, N)
        #     @test size(edge_feature(fg_)) == (out_channel, N)
        #     @test size(positional_feature(fg_)) == (pos_channel, N)

        #     g = gradient(() -> sum(positional_feature(l(fg))), Flux.params(l))
        #     @test length(g.grads) == 2
        # end
    end
end
