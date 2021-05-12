cluster = [1 1 1 1; 2 2 3 3; 4 4 5 5]
X = Array(reshape(1:24, 2, 3, 4))

ys = [3. 3. 4. 4. 5.;
      5. 5. 6. 6. 7.]
us = 2*ones(2, 3, 4)
xs = [1 2 3 4;
      4 2 1 3;
      3 5 5 3]

∇y_mul = [4. 4. 16. 4. 4.; 4. 4. 16. 4. 4.]
∇y_div = [.25 .25 .0625 .25 .25; .25 .25 .0625 .25 .25]
∇u_mean = cat([.5 .5 .25; .5 .5 .25], [.5 .5 .5; .5 .5 .5],
              [.25 .5 .5; .25 .5 .5], [.5 .25 .25; .5 .25 .25], dims=3)

@testset "scatter" begin
    for T = [UInt32, UInt64]
        @testset "$(T)" begin
            @testset "+" begin
                y = [40, 44, 12, 14, 36, 38, 16, 18, 40, 42]
                @test GeometricFlux.scatter(+, cluster, X) == reshape(y, 2, 5)
            end

            @testset "max" begin
                y = [19, 20, 9, 10, 21, 22, 11, 12, 23, 24]
                @test GeometricFlux.scatter(max, cluster, X) == reshape(y, 2, 5)
            end

            @testset "min" begin
                y = [1, 2, 3, 4, 15, 16, 5, 6, 17, 18]
                @test GeometricFlux.scatter(min, cluster, X) == reshape(y, 2, 5)
            end
        end
    end


    for T = [Int32, Int64]
        @testset "$(T)" begin
            @testset "+" begin
                y = [40, 44, 12, 14, 36, 38, 16, 18, 40, 42]
                @test GeometricFlux.scatter(+, cluster, X) == reshape(y, 2, 5)
            end

            @testset "-" begin
                y = [-40, -44, -12, -14, -36, -38, -16, -18, -40, -42]
                @test GeometricFlux.scatter(-, cluster, X) == reshape(y, 2, 5)
            end

            @testset "max" begin
                y = [19, 20, 9, 10, 21, 22, 11, 12, 23, 24]
                @test GeometricFlux.scatter(max, cluster, X) == reshape(y, 2, 5)
            end

            @testset "min" begin
                y = [1, 2, 3, 4, 15, 16, 5, 6, 17, 18]
                @test GeometricFlux.scatter(min, cluster, X) == reshape(y, 2, 5)
            end
        end
    end

    for T = [Float16, Float32, Float64]
        @testset "$(T)" begin
            @testset "+" begin
                y = [40, 44, 12, 14, 36, 38, 16, 18, 40, 42]
                @test GeometricFlux.scatter(+, cluster, X) == reshape(y, 2, 5)
                @test Zygote.gradient(x -> sum(GeometricFlux.scatter(+, x, us)), xs) == (nothing,)
                @test Zygote.gradient(x -> sum(GeometricFlux.scatter(+, xs, x)), us) == (ones(2, 3, 4),)
            end

            @testset "-" begin
                y = [-40, -44, -12, -14, -36, -38, -16, -18, -40, -42]
                @test GeometricFlux.scatter(-, cluster, X) == reshape(y, 2, 5)
                @test Zygote.gradient(x -> sum(GeometricFlux.scatter(-, x, us)), xs) == (nothing,)
                @test Zygote.gradient(x -> sum(GeometricFlux.scatter(-, xs, x)), us) == (-ones(2, 3, 4),)
            end

            @testset "max" begin
                y = [19, 20, 9, 10, 21, 22, 11, 12, 23, 24]
                @test GeometricFlux.scatter(max, cluster, X) == reshape(y, 2, 5)
                @test Zygote.gradient(x -> sum(GeometricFlux.scatter(max, x, us)), xs) == (nothing,)
                @test Zygote.gradient(x -> sum(GeometricFlux.scatter(max, xs, x)), us) == (ones(2, 3, 4),)
            end

            @testset "min" begin
                y = [1, 2, 3, 4, 15, 16, 5, 6, 17, 18]
                @test GeometricFlux.scatter(min, cluster, X) == reshape(y, 2, 5)
                @test Zygote.gradient(x -> sum(GeometricFlux.scatter(min, x, us)), xs) == (nothing,)
                @test Zygote.gradient(x -> sum(GeometricFlux.scatter(min, xs, x)), us) == (ones(2, 3, 4),)
            end

            @testset "*" begin
                y = [1729, 4480, 27, 40, 315, 352, 55, 72, 391, 432]
                @test GeometricFlux.scatter(*, cluster, X) == reshape(y, 2, 5)
                @test Zygote.gradient(x -> sum(GeometricFlux.scatter(*, x, us)), xs) == (nothing,)
                @test Zygote.gradient(x -> sum(GeometricFlux.scatter(*, xs, x)), us) == (2048*ones(2, 3, 4),)
            end

            @testset "/" begin
                y = 1 ./ [1729, 4480, 27, 40, 315, 352, 55, 72, 391, 432]
                @test GeometricFlux.scatter(/, cluster, X) ≈ reshape(y, 2, 5)
                @test Zygote.gradient(x -> sum(GeometricFlux.scatter(/, x, us)), xs) == (nothing,)
                @test Zygote.gradient(x -> sum(GeometricFlux.scatter(/, xs, x)), us) == (-ones(2, 3, 4)/8192,)
            end

            @testset "mean" begin
                y = [10., 11., 6., 7., 18., 19., 8., 9., 20., 21.]
                @test GeometricFlux.scatter(mean, cluster, X) == reshape(y, 2, 5)
                @test Zygote.gradient(x -> sum(GeometricFlux.scatter(mean, x, us)), xs) == (nothing,)
                @test Zygote.gradient(x -> sum(GeometricFlux.scatter(mean, xs, x)), us) == (∇u_mean,)
            end
        end
    end
end
