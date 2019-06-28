@testset "Test GCNConv layer" begin
    adj = [0 1 0 1;
           1 0 1 0;
           0 1 0 1;
           1 0 1 0]
    gc = GCNConv(adj, 3=>5)
    X = rand(4, 3)
    Y = gc(X)
    @test size(Y) == (4, 5)
end
