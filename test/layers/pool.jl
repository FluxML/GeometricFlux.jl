cluster = [1 1 1 1; 2 2 3 3; 4 4 5 5]
X = Array(reshape(1:24, 2, 3, 4))

@testset "Test sumpool" begin
    y = [40, 44, 12, 14, 36, 38, 16, 18, 40, 42]
    @test sumpool(cluster, X) == reshape(y, 2, 5)
end

@testset "Test prodpool" begin
    y = [1729, 4480, 27, 40, 315, 352, 55, 72, 391, 432]
    @test prodpool(cluster, X) == reshape(y, 2, 5)
end

@testset "Test maxpool" begin
    y = [19, 20, 9, 10, 21, 22, 11, 12, 23, 24]
    @test maxpool(cluster, X) == reshape(y, 2, 5)
end

@testset "Test meanpool" begin
    y = [10., 11., 6., 7., 18., 19., 8., 9., 20., 21.]
    @test meanpool(cluster, X) == reshape(y, 2, 5)
end
