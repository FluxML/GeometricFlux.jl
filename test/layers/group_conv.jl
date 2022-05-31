@testset "group conv" begin
    T = Float32
    batch_size = 10
    in_channel = 3
    out_channel = 5

    N = 4
    E = 4
    adj = T[0. 1. 0. 1.;
            1. 0. 1. 0.;
            0. 1. 0. 1.;
            1. 0. 1. 0.]
    fg = FeaturedGraph(adj)

    @testset "EEquivGraphConv" begin
        @testset "layer without static graph" begin
            int_dim = 5
            m_len = in_channel * 2 + 2

            nn_edge = Flux.Dense(m_len, int_dim)
            nn_x = Flux.Dense(int_dim, 1)
            nn_h = Flux.Dense(in_channel + int_dim, out_channel)
            egnn = EEquivGraphConv(in_channel, nn_edge, nn_x, nn_h)

            nf = rand(T, in_channel + 3, N)
            fg = FeaturedGraph(adj, nf=nf)
            fg_ = egnn(fg)
        end
    end
end
