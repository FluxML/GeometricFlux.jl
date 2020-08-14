in_channel = 3
out_channel = 5
N = 6

ug = SimpleGraph(N)
add_edge!(ug, 1, 2); add_edge!(ug, 1, 3); add_edge!(ug, 2, 3)
add_edge!(ug, 3, 4); add_edge!(ug, 2, 5); add_edge!(ug, 3, 6)

dg = SimpleDiGraph(N)
add_edge!(dg, 1, 3); add_edge!(dg, 2, 3); add_edge!(dg, 1, 6)
add_edge!(dg, 2, 5); add_edge!(dg, 3, 4); add_edge!(dg, 3, 5)

el_ug = Vector{Int64}[[2, 3], [1, 3, 5], [1, 2, 4, 6], [3], [2], [3]]
el_dg = Vector{Int64}[[3, 6], [3, 5], [4, 5], [], [], []]

@testset "simplegraphs" begin
    @testset "GCNConv" begin
        gc = GCNConv(ug, in_channel=>out_channel)
        @test size(gc.weight) == (out_channel, in_channel)
        @test size(gc.bias) == (out_channel,)
        @test graph(gc.fg) === ug
    end

    @testset "ChebConv" begin
        k = 4
        cc = ChebConv(ug, in_channel=>out_channel, k)
        @test size(cc.weight) == (out_channel, in_channel, k)
        @test size(cc.bias) == (out_channel,)
        @test graph(cc.fg) == ug
        @test cc.k == k
        @test cc.in_channel == in_channel
        @test cc.out_channel == out_channel
    end

    @testset "GraphConv" begin
        gc = GraphConv(ug, in_channel=>out_channel)
        @test graph(gc.fg) == ug
        @test size(gc.weight1) == (out_channel, in_channel)
        @test size(gc.weight2) == (out_channel, in_channel)
        @test size(gc.bias) == (out_channel,)
    end

    @testset "GATConv" begin
        for heads in [1, 5], concat = [true, false]
            gat = GATConv(ug, in_channel=>out_channel, heads=heads, concat=concat)
            @test graph(gat.fg) == ug
            @test size(gat.weight) == (out_channel * heads, in_channel)
            @test size(gat.bias) == (out_channel * heads,)
            @test size(gat.a) == (2*out_channel, heads, 1)
        end
    end

    @testset "GatedGraphConv" begin
        num_layers = 3
        ggc = GatedGraphConv(ug, out_channel, num_layers)
        @test graph(ggc.fg) == ug
        @test size(ggc.weight) == (out_channel, out_channel, num_layers)
    end

    @testset "EdgeConv" begin
        ec = EdgeConv(ug, Dense(2*in_channel, out_channel))
        @test graph(ec.fg) == ug
    end
end
