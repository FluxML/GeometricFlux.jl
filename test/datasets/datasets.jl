tests = [
    "planetoid",
    "cora",
    "ppi",
]

@testset "datasets" begin
    for t in tests
        include("$(t).jl")
    end
end