@testset "positional" begin
    T = Float32
    batch_size = 10
    in_dim = 3
    in_edge_dim = 2
    pos_dim = 3
    out_dim = 5

    N = 4
    E = 4
    adj = T[0. 1. 0. 1.;
            1. 0. 1. 0.;
            0. 1. 0. 1.;
            1. 0. 1. 0.]
    fg = FeaturedGraph(adj)

    @testset "EEquivGraphPE" begin
        @testset "layer without graph" begin
            l = EEquivGraphPE(in_edge_dim=>out_dim)

            nf = rand(T, out_dim, N)
            ef = rand(T, in_edge_dim, E)
            fg = FeaturedGraph(adj, nf=nf, ef=ef)
            fg_ = l(fg)
            @test size(node_feature(fg_)) == (out_dim, N)

            g = gradient(() -> sum(node_feature(l(fg))), Flux.params(l))
            @test length(g.grads) == 4
        end

        @testset "layer with static graph" begin
            nf = rand(T, out_dim, N, batch_size)
            ef = rand(T, in_edge_dim, E, batch_size)
            l = WithGraph(fg, EEquivGraphPE(in_edge_dim=>out_dim))
            Y = l(nf, ef)
            @test size(Y) == (out_dim, N, batch_size)

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

        g = gradient(() -> sum(positional_feature(l(fg))), Flux.params(l))
        @test length(g.grads) == 2
    end

    @testset "GatedGCNLSPEConv" begin
        @testset "layer without graph" begin
            l = GatedGCNLSPEConv(in_dim=>in_dim, pos_dim, residual=true)

            nf = rand(T, in_dim, N)
            ef = rand(T, in_dim, E)
            pf = rand(T, pos_dim, N)
            fg = FeaturedGraph(adj, nf=nf, ef=ef, pf=pf)
            fg_ = l(fg)
            @test size(node_feature(fg_)) == (in_dim, N)
            @test size(edge_feature(fg_)) == (in_dim, E)
            @test size(positional_feature(fg_)) == (pos_dim, N)

            g = gradient(() -> sum(positional_feature(l(fg))), Flux.params(l))
            @test length(g.grads) == 16
        end

        @testset "layer with static graph" begin
            nf = rand(T, in_dim, N, batch_size)
            ef = rand(T, in_dim, E, batch_size)
            pf = rand(T, pos_dim, N, batch_size)
            l = GatedGCNLSPEConv(in_dim=>in_dim, pos_dim, residual=true)
            l = WithGraph(FeaturedGraph(adj), l)
            H_, E_, X_ = l(nf, ef, pf)
            @test size(H_) == (in_dim, N, batch_size)
            @test size(E_) == (in_dim, N, batch_size)
            @test size(X_) == (pos_dim, N, batch_size)

            g = gradient(() -> sum(l(nf, ef, pf)[1]), Flux.params(l))
            @test length(g.grads) == 14
        end
    end
end
