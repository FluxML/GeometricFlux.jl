using Documenter
using DocumenterCitations
using MultiDocumenter
using DemoCards
using GeometricFlux

const ASSETS = ["assets/flux.css", "assets/favicon.ico"]

bib = CitationBibliography(joinpath(@__DIR__, "bibliography.bib"))

DocMeta.setdocmeta!(GeometricFlux, :DocTestSetup, :(using GeometricFlux, GraphSignals, Flux); recursive=true)

# DemoCards
demopage, postprocess_cb, demo_assets = makedemos("tutorials")
isnothing(demo_assets) || (push!(ASSETS, demo_assets))

# build GeometricFlux documentation
makedocs(
    bib,
    sitename = "GeometricFlux.jl",
    format = Documenter.HTML(
      assets = ASSETS,
      canonical = "https://fluxml.ai/GeometricFlux.jl/stable/",
      analytics = "G-M61P0B2Y8E",
      edit_link = "master",
    ),
    clean = false,
    modules = [GeometricFlux, GraphSignals],
    pages = ["Home" => "index.md",
             demopage,
             "Introduction" => "introduction.md",
             "Basics" => [
                 "Graph Convolutions" => "basics/conv.md",
                 "Graph Passing" => "basics/passgraph.md",
                 "Building Layers" => "basics/layers.md",
                 "Subgraph" => "basics/subgraph.md",
                 "Neighborhood graphs" => "basics/neighborhood_graph.md",
                 "Random graphs" => "basics/random_graph.md",
                 "Batch Learning" => "basics/batch.md",
                ],
             "Cooperate with Flux Layers" => "cooperate.md",
             "Abstractions" => [
               "Message passing scheme" => "abstractions/msgpass.md",
               "Graph network block" => "abstractions/gn.md"],
             "Dynamic Graph Update" => "dynamicgraph.md",
             "Manual" => [
               "FeaturedGraph" => "manual/featuredgraph.md",
               "Graph Convolutional Layers" => "manual/graph_conv.md",
               "Graph Pooling Layers" => "manual/pool.md",
               "Group Convolutional Layers" => "manual/group_conv.md",
               "Positional Encoding Layers" => "manual/positional.md",
               "Embeddings" => "manual/embedding.md",
               "Models" => "manual/models.md",
               "Linear Algebra" => "manual/linalg.md",
               "Neighborhood graphs" => "manual/neighborhood_graph.md",
               ],
             "References" => "references.md",
    ]
)

# callbacks of DemoCards
postprocess_cb()

# build MultiDocumenter
clonedir = mktempdir()

docs = [
    ("FluxML/GeometricFlux.jl.git", "gh-pages", false) => MultiDocumenter.MultiDocRef(
        upstream = joinpath(clonedir, "GeometricFlux"),
        path = "geometricflux",
        name = "GeometricFlux"
    ),
    ("yuehhua/GraphSignals.jl.git", "gh-pages", false) => MultiDocumenter.MultiDocRef(
        upstream = joinpath(clonedir, "GraphSignals"),
        path = "graphsignals",
        name = "GraphSignals"
    ),
]

for ((remote, branch, use_ssh), docref) in docs
  prefix = use_ssh ? "git@github.com:" : "https://github.com/"
  run(`git clone --depth 1 $prefix$remote --branch $branch --single-branch $(docref.upstream)`)
end

outpath = joinpath(@__DIR__, "out")

MultiDocumenter.make(
  outpath,
  collect(last.(docs));
  search_engine = MultiDocumenter.SearchConfig(
      index_versions = ["stable"],
      engine = MultiDocumenter.FlexSearch
  )
)

deploydocs(
  repo = "github.com/FluxML/GeometricFlux.jl.git",
  target = "out",
)
