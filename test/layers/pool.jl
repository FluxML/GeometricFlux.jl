cluster = [1 1 1 1; 2 2 3 3; 4 4 5 5]
X = Array(reshape(1:24, 2, 3, 4))

@testset "pool" begin
    for T = [UInt32, UInt64]
        @testset "$(T)" begin
            @testset "sumpool" begin
                y = [40, 44, 12, 14, 36, 38, 16, 18, 40, 42]
                @test sumpool(cluster, X) == reshape(y, 2, 5)
                @test pool(:add, cluster, X) == reshape(y, 2, 5)
            end

            @testset "maxpool" begin
                y = [19, 20, 9, 10, 21, 22, 11, 12, 23, 24]
                @test maxpool(cluster, X) == reshape(y, 2, 5)
                @test pool(:max, cluster, X) == reshape(y, 2, 5)
            end

            @testset "minpool" begin
                y = [1, 2, 3, 4, 15, 16, 5, 6, 17, 18]
                @test minpool(cluster, X) == reshape(y, 2, 5)
                @test pool(:min, cluster, X) == reshape(y, 2, 5)
            end
        end
    end


    for T = [Int32, Int64]
        @testset "$(T)" begin
            @testset "sumpool" begin
                y = [40, 44, 12, 14, 36, 38, 16, 18, 40, 42]
                @test sumpool(cluster, X) == reshape(y, 2, 5)
                @test pool(:add, cluster, X) == reshape(y, 2, 5)
            end

            @testset "subpool" begin
                y = [-40, -44, -12, -14, -36, -38, -16, -18, -40, -42]
                @test subpool(cluster, X) == reshape(y, 2, 5)
                @test pool(:sub, cluster, X) == reshape(y, 2, 5)
            end

            @testset "maxpool" begin
                y = [19, 20, 9, 10, 21, 22, 11, 12, 23, 24]
                @test maxpool(cluster, X) == reshape(y, 2, 5)
                @test pool(:max, cluster, X) == reshape(y, 2, 5)
            end

            @testset "minpool" begin
                y = [1, 2, 3, 4, 15, 16, 5, 6, 17, 18]
                @test minpool(cluster, X) == reshape(y, 2, 5)
                @test pool(:min, cluster, X) == reshape(y, 2, 5)
            end
        end
    end

    for T = [Float16, Float32, Float64]
        @testset "$(T)" begin
            @testset "sumpool" begin
                y = [40, 44, 12, 14, 36, 38, 16, 18, 40, 42]
                @test sumpool(cluster, X) == reshape(y, 2, 5)
                @test pool(:add, cluster, X) == reshape(y, 2, 5)
            end

            @testset "subpool" begin
                y = [-40, -44, -12, -14, -36, -38, -16, -18, -40, -42]
                @test subpool(cluster, X) == reshape(y, 2, 5)
                @test pool(:sub, cluster, X) == reshape(y, 2, 5)
            end

            @testset "maxpool" begin
                y = [19, 20, 9, 10, 21, 22, 11, 12, 23, 24]
                @test maxpool(cluster, X) == reshape(y, 2, 5)
                @test pool(:max, cluster, X) == reshape(y, 2, 5)
            end

            @testset "minpool" begin
                y = [1, 2, 3, 4, 15, 16, 5, 6, 17, 18]
                @test minpool(cluster, X) == reshape(y, 2, 5)
                @test pool(:min, cluster, X) == reshape(y, 2, 5)
            end

            @testset "prodpool" begin
                y = [1729, 4480, 27, 40, 315, 352, 55, 72, 391, 432]
                @test prodpool(cluster, X) == reshape(y, 2, 5)
                @test pool(:mul, cluster, X) == reshape(y, 2, 5)
            end

            @testset "divpool" begin
                y = 1 ./ [1729, 4480, 27, 40, 315, 352, 55, 72, 391, 432]
                @test divpool(cluster, X) ≈ reshape(y, 2, 5)
                @test pool(:div, cluster, X) ≈ reshape(y, 2, 5)
            end

            @testset "meanpool" begin
                y = [10., 11., 6., 7., 18., 19., 8., 9., 20., 21.]
                @test meanpool(cluster, X) == reshape(y, 2, 5)
                @test pool(:mean, cluster, X) == reshape(y, 2, 5)
            end
        end
    end
end
