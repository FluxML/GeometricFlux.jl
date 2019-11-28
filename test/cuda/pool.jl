using CuArrays

cluster = [1 1 1 1; 2 2 3 3; 4 4 5 5]
X = CuArray(reshape(1:24, 2, 3, 4))

@testset "cuda/pool" begin
    for T = [UInt32, UInt64]
        @testset "$(T)" begin
            @testset "sumpool" begin
                y = [40, 44, 12, 14, 36, 38, 16, 18, 40, 42]
                y = reshape(y, 2, 5)
                @test sumpool(CuArray{Int64}(cluster), T.(X)) == T.(y)
                @test pool(:add, CuArray{Int64}(cluster), T.(X)) == T.(y)
                @test sumpool(cluster, T.(X)) == T.(y)
                @test pool(:add, cluster, T.(X)) == T.(y)
            end

            @testset "maxpool" begin
                y = [19, 20, 9, 10, 21, 22, 11, 12, 23, 24]
                y = reshape(y, 2, 5)
                @test maxpool(CuArray{Int64}(cluster), T.(X)) == T.(y)
                @test pool(:max, CuArray{Int64}(cluster), T.(X)) == T.(y)
                @test maxpool(cluster, T.(X)) == T.(y)
                @test pool(:max, cluster, T.(X)) == T.(y)
            end

            @testset "minpool" begin
                y = [1, 2, 3, 4, 15, 16, 5, 6, 17, 18]
                y = reshape(y, 2, 5)
                @test minpool(CuArray{Int64}(cluster), T.(X)) == T.(y)
                @test pool(:min, CuArray{Int64}(cluster), T.(X)) == T.(y)
                @test minpool(cluster, T.(X)) == T.(y)
                @test pool(:min, cluster, T.(X)) == T.(y)
            end
        end
    end

    for T = [Int32, Int64]
        @testset "$(T)" begin
            @testset "sumpool" begin
                y = [40, 44, 12, 14, 36, 38, 16, 18, 40, 42]
                y = reshape(y, 2, 5)
                @test sumpool(CuArray{Int64}(cluster), T.(X)) == T.(y)
                @test pool(:add, CuArray{Int64}(cluster), T.(X)) == T.(y)
                @test sumpool(cluster, T.(X)) == T.(y)
                @test pool(:add, cluster, T.(X)) == T.(y)
            end

            @testset "subpool" begin
                y = [-40, -44, -12, -14, -36, -38, -16, -18, -40, -42]
                y = reshape(y, 2, 5)
                @test subpool(CuArray{Int64}(cluster), T.(X)) == T.(y)
                @test pool(:sub, CuArray{Int64}(cluster), T.(X)) == T.(y)
                @test subpool(cluster, T.(X)) == T.(y)
                @test pool(:sub, cluster, T.(X)) == T.(y)
            end

            @testset "maxpool" begin
                y = [19, 20, 9, 10, 21, 22, 11, 12, 23, 24]
                y = reshape(y, 2, 5)
                @test maxpool(CuArray{Int64}(cluster), T.(X)) == T.(y)
                @test pool(:max, CuArray{Int64}(cluster), T.(X)) == T.(y)
                @test maxpool(cluster, T.(X)) == T.(y)
                @test pool(:max, cluster, T.(X)) == T.(y)
            end

            @testset "minpool" begin
                y = [1, 2, 3, 4, 15, 16, 5, 6, 17, 18]
                y = reshape(y, 2, 5)
                @test minpool(CuArray{Int64}(cluster), T.(X)) == T.(y)
                @test pool(:min, CuArray{Int64}(cluster), T.(X)) == T.(y)
                @test minpool(cluster, T.(X)) == T.(y)
                @test pool(:min, cluster, T.(X)) == T.(y)
            end
        end
    end

    for T = [Float32, Float64]
        @testset "$(T)" begin
            @testset "sumpool" begin
                y = [40, 44, 12, 14, 36, 38, 16, 18, 40, 42]
                y = reshape(y, 2, 5)
                @test sumpool(CuArray{Int64}(cluster), T.(X)) == T.(y)
                @test pool(:add, CuArray{Int64}(cluster), T.(X)) == T.(y)
                @test sumpool(cluster, T.(X)) == T.(y)
                @test pool(:add, cluster, T.(X)) == T.(y)
            end

            @testset "subpool" begin
                y = [-40, -44, -12, -14, -36, -38, -16, -18, -40, -42]
                y = reshape(y, 2, 5)
                @test subpool(CuArray{Int64}(cluster), T.(X)) == T.(y)
                @test pool(:sub, CuArray{Int64}(cluster), T.(X)) == T.(y)
                @test subpool(cluster, T.(X)) == T.(y)
                @test pool(:sub, cluster, T.(X)) == T.(y)
            end

            @testset "maxpool" begin
                y = [19, 20, 9, 10, 21, 22, 11, 12, 23, 24]
                y = reshape(y, 2, 5)
                @test maxpool(CuArray{Int64}(cluster), T.(X)) == T.(y)
                @test pool(:max, CuArray{Int64}(cluster), T.(X)) == T.(y)
                @test maxpool(cluster, T.(X)) == T.(y)
                @test pool(:max, cluster, T.(X)) == T.(y)
            end

            @testset "minpool" begin
                y = [1, 2, 3, 4, 15, 16, 5, 6, 17, 18]
                y = reshape(y, 2, 5)
                @test minpool(CuArray{Int64}(cluster), T.(X)) == T.(y)
                @test pool(:min, CuArray{Int64}(cluster), T.(X)) == T.(y)
                @test minpool(cluster, T.(X)) == T.(y)
                @test pool(:min, cluster, T.(X)) == T.(y)
            end

            @testset "prodpool" begin
                y = [1729, 4480, 27, 40, 315, 352, 55, 72, 391, 432]
                y = reshape(y, 2, 5)
                @test prodpool(CuArray{Int64}(cluster), T.(X)) == T.(y)
                @test pool(:mul, CuArray{Int64}(cluster), T.(X)) == T.(y)
                @test prodpool(cluster, T.(X)) == T.(y)
                @test pool(:mul, cluster, T.(X)) == T.(y)
            end

            @testset "divpool" begin
                y = 1 ./ [1729, 4480, 27, 40, 315, 352, 55, 72, 391, 432]
                y = reshape(y, 2, 5)
                @test divpool(CuArray{Int64}(cluster), T.(X)) ≈ T.(y)
                @test pool(:div, CuArray{Int64}(cluster), T.(X)) ≈ T.(y)
                @test divpool(cluster, T.(X)) ≈ T.(y)
                @test pool(:div, cluster, T.(X)) ≈ T.(y)
            end

            @testset "meanpool" begin
                y = [10., 11., 6., 7., 18., 19., 8., 9., 20., 21.]
                y = reshape(y, 2, 5)
                @test meanpool(CuArray{Int64}(cluster), T.(X)) == T.(y)
                @test pool(:mean, CuArray{Int64}(cluster), T.(X)) == T.(y)
                @test meanpool(cluster, T.(X)) == T.(y)
                @test pool(:mean, cluster, T.(X)) == T.(y)
            end
        end
    end
end
