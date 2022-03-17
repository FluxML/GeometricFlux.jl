using Documenter
using DocumenterCitations
using GeometricFlux

bib = CitationBibliography(joinpath(@__DIR__, "bibliography.bib"), sorting=:nyt)

DocMeta.setdocmeta!(GeometricFlux, :DocTestSetup, :(using GeometricFlux, Flux); recursive=true)

makedocs(
    bib,
    sitename = "GeometricFlux.jl",
    format = Documenter.HTML(
      assets = ["assets/flux.css"],
      canonical = "https://fluxml.ai/GeometricFlux.jl/stable/",
      analytics = "G-M61P0B2Y8E",
    ),
    clean = false,
    modules = [GeometricFlux,GraphSignals],
    pages = ["Home" => "index.md",
             "Introduction" => "introduction.md",
             "Basics" => [
                 "Graph Convolutions" => "basics/conv.md",
                 "Graph Passing" => "basics/passgraph.md",
                 "Building Layers" => "basics/layers.md",
                 "Subgraph" => "basics/subgraph.md",
                 "Batch Learning" => "basics/batch.md",
                ],
             "Cooperate with Flux Layers" => "cooperate.md",
             "Tutorials" => [
                  "Semi-Supervised Learning with GCN" => "tutorials/semisupervised_gcn.md",
                  "GCN with Fixed Graph" => "tutorials/gcn_fixed_graph.md",
                ],
             "Abstractions" => [
               "Message passing scheme" => "abstractions/msgpass.md",
               "Graph network block" => "abstractions/gn.md"],
             "Manual" => [
               "FeaturedGraph" => "manual/featuredgraph.md",
               "Convolutional Layers" => "manual/conv.md",
               "Pooling Layers" => "manual/pool.md",
               "Embeddings" => "manual/embedding.md",
               "Models" => "manual/models.md",
               "Linear Algebra" => "manual/linalg.md"
               ],
             "References" => "references.md",
    ]
)

deploydocs(
  repo = "github.com/FluxML/GeometricFlux.jl.git",
  target = "build",
)
