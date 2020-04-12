using Documenter
using GeometricFlux

makedocs(
    sitename = "GeometricFlux",
    format = Documenter.HTML(),
    modules = [GeometricFlux],
    pages = ["Home" => "index.md",
             "Manual" =>
               ["Convolutional Layers" => "manual/conv.md",
                "Pooling Layers" => "manual/pool.md",
                "Models" => "manual/models.md",
                "Linear Algebra" => "manual/linalg.md",
                "Utilities" => "manual/utils.md"]
    ]
)

deploydocs(repo = "github.com/yuehhua/GeometricFlux.jl.git")
