@testset "positional" begin
    T = Float32
    batch_size = 10
    in_channel = 3
    in_channel_edge = 2
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

            g = Zygote.gradient(() -> sum(node_feature(l(fg))), Flux.params(l))
            @test length(g.grads) == 4
        end

        @testset "layer with static graph" begin
            nf = rand(T, out_channel, N, batch_size)
            ef = rand(T, in_channel_edge, E, batch_size)
            l = WithGraph(fg, EEquivGraphPE(in_channel_edge=>out_channel))
            Y = l(nf, ef)
            @test size(Y) == (out_channel, N, batch_size)

            g = Zygote.gradient(() -> sum(l(nf, ef)), Flux.params(l))
            @test length(g.grads) == 2
        end
    end
end
