cluster = [1 1 1 1; 2 2 3 3; 4 4 5 5]
X = CuArray(reshape(1:24, 2, 3, 4))

@testset "cuda/scatter" begin
    for T = [UInt32, UInt64]
        @testset "$(T)" begin
            @testset "+" begin
                y = [40, 44, 12, 14, 36, 38, 16, 18, 40, 42]
                y = reshape(y, 2, 5)
                @test GeometricFlux.scatter(+, CuArray{Int64}(cluster), T.(X)) == T.(y)
                @test GeometricFlux.scatter(+, cluster, T.(X)) == T.(y)
            end

            @testset "max" begin
                y = [19, 20, 9, 10, 21, 22, 11, 12, 23, 24]
                y = reshape(y, 2, 5)
                @test GeometricFlux.scatter(max, CuArray{Int64}(cluster), T.(X)) == T.(y)
                @test GeometricFlux.scatter(max, cluster, T.(X)) == T.(y)
            end

            @testset "min" begin
                y = [1, 2, 3, 4, 15, 16, 5, 6, 17, 18]
                y = reshape(y, 2, 5)
                @test GeometricFlux.scatter(min, CuArray{Int64}(cluster), T.(X)) == T.(y)
                @test GeometricFlux.scatter(min, cluster, T.(X)) == T.(y)
            end
        end
    end

    for T = [Int32, Int64]
        @testset "$(T)" begin
            @testset "+" begin
                y = [40, 44, 12, 14, 36, 38, 16, 18, 40, 42]
                y = reshape(y, 2, 5)
                @test GeometricFlux.scatter(+, CuArray{Int64}(cluster), T.(X)) == T.(y)
                @test GeometricFlux.scatter(+, cluster, T.(X)) == T.(y)
            end

            @testset "-" begin
                y = [-40, -44, -12, -14, -36, -38, -16, -18, -40, -42]
                y = reshape(y, 2, 5)
                @test GeometricFlux.scatter(-, CuArray{Int64}(cluster), T.(X)) == T.(y)
                @test GeometricFlux.scatter(-, cluster, T.(X)) == T.(y)
            end

            @testset "max" begin
                y = [19, 20, 9, 10, 21, 22, 11, 12, 23, 24]
                y = reshape(y, 2, 5)
                @test GeometricFlux.scatter(max, CuArray{Int64}(cluster), T.(X)) == T.(y)
                @test GeometricFlux.scatter(max, cluster, T.(X)) == T.(y)
            end

            @testset "min" begin
                y = [1, 2, 3, 4, 15, 16, 5, 6, 17, 18]
                y = reshape(y, 2, 5)
                @test GeometricFlux.scatter(min, CuArray{Int64}(cluster), T.(X)) == T.(y)
                @test GeometricFlux.scatter(min, cluster, T.(X)) == T.(y)
            end
        end
    end

    for T = [Float32, Float64]
        @testset "$(T)" begin
            @testset "+" begin
                y = [40, 44, 12, 14, 36, 38, 16, 18, 40, 42]
                y = reshape(y, 2, 5)
                @test GeometricFlux.scatter(+, CuArray{Int64}(cluster), T.(X)) == T.(y)
                @test GeometricFlux.scatter(+, cluster, T.(X)) == T.(y)
            end

            @testset "-" begin
                y = [-40, -44, -12, -14, -36, -38, -16, -18, -40, -42]
                y = reshape(y, 2, 5)
                @test GeometricFlux.scatter(-, CuArray{Int64}(cluster), T.(X)) == T.(y)
                @test GeometricFlux.scatter(-, cluster, T.(X)) == T.(y)
            end

            @testset "max" begin
                y = [19, 20, 9, 10, 21, 22, 11, 12, 23, 24]
                y = reshape(y, 2, 5)
                @test GeometricFlux.scatter(max, CuArray{Int64}(cluster), T.(X)) == T.(y)
                @test GeometricFlux.scatter(max, cluster, T.(X)) == T.(y)
            end

            @testset "min" begin
                y = [1, 2, 3, 4, 15, 16, 5, 6, 17, 18]
                y = reshape(y, 2, 5)
                @test GeometricFlux.scatter(min, CuArray{Int64}(cluster), T.(X)) == T.(y)
                @test GeometricFlux.scatter(min, cluster, T.(X)) == T.(y)
            end

            @testset "*" begin
                y = [1729, 4480, 27, 40, 315, 352, 55, 72, 391, 432]
                y = reshape(y, 2, 5)
                @test GeometricFlux.scatter(*, CuArray{Int64}(cluster), T.(X)) == T.(y)
                @test GeometricFlux.scatter(*, cluster, T.(X)) == T.(y)
            end

            @testset "/" begin
                # It seems that y have to convert to CuArray to avoid error,
                # instead of broadcastly casting an array
                y = 1 ./ [1729, 4480, 27, 40, 315, 352, 55, 72, 391, 432]
                y = reshape(y, 2, 5)
                @test GeometricFlux.scatter(/, CuArray{Int64}(cluster), T.(X)) ≈ CuArray{T}(y)
                @test GeometricFlux.scatter(/, cluster, T.(X)) ≈ CuArray{T}(y)
            end

            @testset "mean" begin
                y = [10., 11., 6., 7., 18., 19., 8., 9., 20., 21.]
                y = reshape(y, 2, 5)
                @test GeometricFlux.scatter(mean, CuArray{Int64}(cluster), T.(X)) == T.(y)
                @test GeometricFlux.scatter(mean, cluster, T.(X)) == T.(y)
            end
        end
    end
end
