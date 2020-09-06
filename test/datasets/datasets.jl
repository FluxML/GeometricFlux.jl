tests = [
    "planetoid",
    "cora",
    "ppi",
    "reddit",
]

@testset "datasets" begin
    for t in tests
        include("$(t).jl")
    end
end