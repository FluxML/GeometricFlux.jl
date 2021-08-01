@testset "FeaturedGraph" begin
    @testset "symmetric graph" begin
        u = [1, 2, 3, 4, 2, 3, 4, 1]
        v = [2, 3, 4, 1, 1, 2, 3, 4]
        adj_mat =  [0  1  0  1
                    1  0  1  0
                    0  1  0  1
                    1  0  1  0]
        adj_list_out =  [[2,4], [1,3], [2,4], [1,3]]
        adj_list_in =  [[2,4], [1,3], [2,4], [1,3]]

        # core functionality
        fg = FeaturedGraph(u, v; graph_type=GRAPH_T)
        @test fg.num_edges == 8
        @test fg.num_nodes == 4
        @test collect(edges(fg)) |> sort == collect(zip(u, v)) |> sort
        @test sort(outneighbors(fg, 1)) == [2, 4] 
        @test sort(inneighbors(fg, 1)) == [2, 4] 
        @test is_directed(fg) == true

        # adjacency
        @test adjacency_matrix(fg) == adj_mat
        @test adjacency_matrix(fg; dir=:in) == adj_mat
        @test adjacency_matrix(fg; dir=:out) == adj_mat
        @test sort.(adjacency_list(fg; dir=:in)) == adj_list_in
        @test sort.(adjacency_list(fg; dir=:out)) == adj_list_out

        @testset "constructors" begin
            fg = FeaturedGraph(adj_mat; graph_type=GRAPH_T)
            adjacency_matrix(fg; dir=:out) == adj_mat
            adjacency_matrix(fg; dir=:in) == adj_mat
        end 

        @testset "degree" begin
            fg = FeaturedGraph(adj_mat; graph_type=GRAPH_T)
            @test degree(fg, dir=:out) == vec(sum(adj_mat, dims=2))
            @test degree(fg, dir=:in) == vec(sum(adj_mat, dims=1))
        end
    end

    @testset "asymmetric graph" begin
        u = [1, 2, 3, 4]
        v = [2, 3, 4, 1]
        adj_mat_out =  [0  1  0  0
                        0  0  1  0
                        0  0  0  1
                        1  0  0  0]
        adj_list_out =  [[2], [3], [4], [1]]


        adj_mat_in =   [0  0  0  1
                        1  0  0  0
                        0  1  0  0
                        0  0  1  0]
        adj_list_in =  [[4], [1], [2], [3]]

        # core functionality
        fg = FeaturedGraph(u, v; graph_type=GRAPH_T)
        @test fg.num_edges == 4
        @test fg.num_nodes == 4
        @test collect(edges(fg)) |> sort == collect(zip(u, v)) |> sort
        @test sort(outneighbors(fg, 1)) == [2] 
        @test sort(inneighbors(fg, 1)) == [4] 
        @test is_directed(fg) == true

        # adjacency
        @test adjacency_matrix(fg) ==  adj_mat_out
        @test adjacency_list(fg) ==  adj_list_out
        @test adjacency_matrix(fg, dir=:out) ==  adj_mat_out
        @test adjacency_list(fg, dir=:out) ==  adj_list_out
        @test adjacency_matrix(fg, dir=:in) ==  adj_mat_in
        @test adjacency_list(fg, dir=:in) ==  adj_list_in

        @testset "degree" begin
            fg = FeaturedGraph(adj_mat_out; graph_type=GRAPH_T)
            @test degree(fg, dir=:out) == vec(sum(adj_mat_out, dims=2))
            @test degree(fg, dir=:in) == vec(sum(adj_mat_out, dims=1))
        end
    end

end