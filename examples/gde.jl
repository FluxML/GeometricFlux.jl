using GeometricFlux, Flux, JLD2, SparseArrays, DiffEqFlux, DifferentialEquations
using Flux: onehotbatch, onecold, logitcrossentropy, throttle
using Flux: @epochs
using Statistics: mean
using LightGraphs: adjacency_matrix
using CUDA

# Load the dataset
@load "data/cora_features.jld2" features
@load "data/cora_labels.jld2" labels
@load "data/cora_graph.jld2" g

# Model and Data Configuration
num_nodes = 2708
num_features = 1433
hidden = 16
target_catg = 7
epochs = 40

# Preprocess the data and compute adjacency matrix
train_X = Matrix{Float32}(features) |> gpu # dim: num_features * num_nodes
train_y = Float32.(labels) |> gpu  # dim: target_catg * num_nodes
adj_mat = Matrix{Float32}(adjacency_matrix(g)) |> gpu 

# Define the Neural GDE
# diffeqarray_to_array(x) = reshape(cpu(x), size(x)[1:2])

# NeuralODE just needs first component to be in gpu()
node = NeuralODE(
    gpu(GCNConv(adj_mat, hidden=>hidden)),
    (0.f0, 1.f0), Tsit5(), save_everystep = false,
    reltol = 1e-3, abstol = 1e-3, save_start = false
)

model = Chain(GCNConv(adj_mat, num_features=>hidden, relu),
              Dropout(0.5),
              node,
              arr -> arr[1],
              GCNConv(adj_mat, hidden=>target_catg),
              softmax) |> gpu

# Loss
loss(x, y) = logitcrossentropy(model(x), y)
accuracy(x, y) = mean(onecold(cpu(model(x))) .== onecold(cpu(y)))

# Training
## Model Parameters
ps = Flux.params(model, node.p);
## Training Data
train_data = [(train_X, train_y)]
## Optimizer
opt = ADAM(0.05)
## Callback Function for printing accuracies
evalcb() = @show(accuracy(train_X, train_y))

## Training Loop
@epochs epochs Flux.train!(loss, ps, train_data, opt, cb=throttle(evalcb, 10))
