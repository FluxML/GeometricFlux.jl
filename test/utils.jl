@testset "gather" begin
    @testset "3-argumented" begin
        X = [1 2 3; 4 5 6]
        ind1 = [1 2 2; 1 1 1]
        ind2 = [1 2; 3 1]
        for T = [Int8, Int16, Int32, Int64, Int128,
                 Float16, Float32, Float64]
            @test gather(T.(X), ind1, 1) == T.([1 5 6; 1 2 3])
            @test gather(T.(X), ind2, 2) == T.([1 2; 6 4])
        end
    end

    @testset "2-argumented" begin
        input = [3 3 4 4 5;
                 5 5 6 6 7]
        index = [1 2 3 4;
                 4 2 1 3;
                 3 5 5 3]
        output = cat([3 4 4; 5 6 6], [3 3 5; 5 5 7],
                     [4 3 5; 6 5 7], [4 4 4; 6 6 6], dims=3)
        @test gather(input, index) == output
    end
end

@testset "GraphInfo" begin
    adjl = [[2,3,4,5], [1,3,4], [2,4], [1,2,3], [1]]
    gi = GraphInfo(adjl)
    @test gi.edge_idx == [0, 4, 7, 9, 12, 13]
    @test gi.V == 5
    @test gi.E == 13
end
