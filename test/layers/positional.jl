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

    @testset "LSPE" begin
        K = 3
        l = LSPE(fg, K)
        @test size(l.pe) == (K, N)

        fg_ = l(fg)
        @test GraphSignals.has_positional_feature(fg_)
    end

    @testset "GatedGCNLSPEConv" begin
        @testset "layer without graph" begin
            l = GatedGCNLSPEConv(in_channel=>in_channel, pos_channel, residual=true)

            nf = rand(T, in_channel, N)
            ef = rand(T, in_channel, E)
            pf = rand(T, pos_channel, N)
            fg = FeaturedGraph(adj, nf=nf, ef=ef, pf=pf)
            fg_ = l(fg)
            @test size(node_feature(fg_)) == (in_channel, N)
            @test size(edge_feature(fg_)) == (in_channel, E)
            @test size(positional_feature(fg_)) == (pos_channel, N)

            g = gradient(() -> sum(positional_feature(l(fg))), Flux.params(l))
            @test length(g.grads) == 16
        end

        @testset "layer with static graph" begin
            nf = rand(T, in_channel, N, batch_size)
            ef = rand(T, in_channel, E, batch_size)
            pf = rand(T, pos_channel, N, batch_size)
            l = GatedGCNLSPEConv(in_channel=>in_channel, pos_channel, residual=true)
            l = WithGraph(FeaturedGraph(adj), l)
            H_, E_, X_ = l(nf, ef, pf)
            @test size(H_) == (in_channel, N, batch_size)
            @test size(E_) == (in_channel, N, batch_size)
            @test size(X_) == (pos_channel, N, batch_size)

            g = gradient(() -> sum(l(nf, ef, pf)[1]), Flux.params(l))
            @test length(g.grads) == 14
        end
    end
end
