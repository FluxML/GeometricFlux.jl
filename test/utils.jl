N = 6
F = 100

@testset "Test neighboring" begin
    X = Array(reshape(1:N*F, F, N))
    neighbors = [[2], [1,4,5,6], [6], [2,5], [2,4,6], [2,3,5]]
    Y, clst = neighboring(X, neighbors)
    @test size(Y) == (100, 14)
    @test Y[1, :] == [101, 1, 301, 401, 501, 501, 101, 401, 101, 301, 501, 101, 201, 401]
    @test clst == [1, 2, 2, 2, 2, 3, 4, 4, 5, 5, 5, 6 ,6, 6]
end
