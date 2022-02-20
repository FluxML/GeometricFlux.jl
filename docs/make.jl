using Documenter
using DocumenterCitations
using GeometricFlux

bib = CitationBibliography(joinpath(@__DIR__, "bibliography.bib"), sorting=:nyt)

makedocs(
    bib,
    sitename = "GeometricFlux.jl",
    format = Documenter.HTML(
      assets = ["assets/flux.css"],
      canonical = "https://fluxml.ai/GeometricFlux.jl/stable/",
      analytics = "G-M61P0B2Y8E",
    ),
    clean = false,
    modules = [GeometricFlux],
    pages = ["Home" => "index.md",
             "Get started" => "start.md",
             "Basics" =>
               ["Graph convolutions" => "basics/conv.md",
                "Building layers" => "basics/layers.md",
                "Graph passing" => "basics/passgraph.md"],
             "Cooperate with Flux layers" => "cooperate.md",
             "Tutorials" =>
                [
                  "Semi-supervised learning with GCN" => "tutorials/semisupervised_gcn.md",
                  "GCN with Fixed Graph" => "tutorials/gcn_fixed_graph.md",
                ],
             "Abstractions" =>
               ["Message passing scheme" => "abstractions/msgpass.md",
                "Graph network block" => "abstractions/gn.md"],
             "Manual" =>
               ["Convolutional Layers" => "manual/conv.md",
                "Pooling Layers" => "manual/pool.md",
                "Models" => "manual/models.md",
                "Linear Algebra" => "manual/linalg.md"],
             "References" => "references.md",
    ]
)

deploydocs(
  repo = "github.com/FluxML/GeometricFlux.jl.git",
  target = "build",
)
