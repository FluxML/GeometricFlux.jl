using SimpleWeightedGraphs
using SimpleWeightedGraphs: add_edge!

el_ug = Vector{Int64}[[3, 6], [3, 5], [1, 2, 4, 5], [3], [2, 3], [1]]
el_dg = Vector{Int64}[[3, 6], [3, 5], [4, 5], [], [], []]

@testset "Test adjlist" begin
    ug = SimpleGraph(6)
    add_edge!(ug, 1, 3); add_edge!(ug, 2, 3); add_edge!(ug, 1, 6)
    add_edge!(ug, 2, 5); add_edge!(ug, 3, 4); add_edge!(ug, 3, 5)
    @test adjlist(ug) == el_ug

    dg = SimpleDiGraph(6)
    add_edge!(dg, 1, 3); add_edge!(dg, 2, 3); add_edge!(dg, 1, 6)
    add_edge!(dg, 2, 5); add_edge!(dg, 3, 4); add_edge!(dg, 3, 5)
    @test adjlist(dg) == el_dg

    ug = SimpleWeightedGraph(6)
    add_edge!(ug, 1, 3, 2); add_edge!(ug, 2, 3, 2); add_edge!(ug, 1, 6, 1)
    add_edge!(ug, 2, 5, -2); add_edge!(ug, 3, 4, -2); add_edge!(ug, 3, 5, -1)
    @test adjlist(ug) == el_ug

    dg = SimpleWeightedDiGraph(6)
    add_edge!(dg, 1, 3, 2); add_edge!(dg, 2, 3, 2); add_edge!(dg, 1, 6, 1)
    add_edge!(dg, 2, 5, -2); add_edge!(dg, 3, 4, -2); add_edge!(dg, 3, 5, -1)
    @test adjlist(dg) == el_dg
end
