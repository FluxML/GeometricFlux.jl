@testset "group conv" begin
    T = Float32
    batch_size = 10
    in_channel = 3
    out_channel = 5
    pos_dim = 2
    in_channel_edge = 7

    N = 4
    E = 4
    adj = T[0. 1. 0. 1.;
            1. 0. 1. 0.;
            0. 1. 0. 1.;
            1. 0. 1. 0.]

    @testset "EEquivGraphConv" begin
        @testset "layer without graph" begin
            egnn = EEquivGraphConv(in_channel=>out_channel, pos_dim, in_channel_edge)

            nf = rand(T, in_channel, N)
            ef = rand(T, in_channel_edge, E)
            pf = rand(T, pos_dim, N)
            fg = FeaturedGraph(adj, nf=nf, ef=ef, pf=pf)
            fg_ = egnn(fg)
            nf_ = node_feature(fg_)
            pf_ = positional_feature(fg_)

            @test size(nf_) == (out_channel, N)
            @test size(pf_) == (pos_dim, N)

            g = gradient(() -> sum(node_feature(egnn(fg))), Flux.params(egnn))
            @test length(g.grads) == 13
        end

        @testset "layer without graph and edge feature" begin
            egnn = EEquivGraphConv(in_channel=>out_channel, pos_dim)

            nf = rand(T, in_channel, N)
            pf = rand(T, pos_dim, N)
            fg = FeaturedGraph(adj, nf=nf, pf=pf)
            fg_ = egnn(fg)
            nf_ = node_feature(fg_)
            pf_ = positional_feature(fg_)

            @test size(nf_) == (out_channel, N)
            @test size(pf_) == (pos_dim, N)

            g = gradient(() -> sum(node_feature(egnn(fg))), Flux.params(egnn))
            @test length(g.grads) == 13
        end

        @testset "layer with static graph" begin
            nf = rand(T, in_channel, N, batch_size)
            ef = rand(T, in_channel_edge, E, batch_size)
            fg = FeaturedGraph(adj, pf = rand(T, pos_dim, N, batch_size))
            l = WithGraph(fg, EEquivGraphConv(in_channel=>out_channel, pos_dim, in_channel_edge))
            H, Y = l(nf, ef)
            @test size(H) == (out_channel, N, batch_size)
            @test size(Y) == (pos_dim, N, batch_size)

            g = gradient(() -> sum(l(nf, ef)[1]), Flux.params(l))
            @test length(g.grads) == 11
        end

        @testset "layer with static graph without edge feature" begin
            nf = rand(T, in_channel, N, batch_size)
            fg = FeaturedGraph(adj, pf = rand(T, pos_dim, N, batch_size))
            l = WithGraph(fg, EEquivGraphConv(in_channel=>out_channel, pos_dim))
            H, Y = l(nf)
            @test size(H) == (out_channel, N, batch_size)
            @test size(Y) == (pos_dim, N, batch_size)

            g = gradient(() -> sum(l(nf)[1]), Flux.params(l))
            @test length(g.grads) == 11
        end
    end
end
