using GeometricFlux, Flux, JLD2, SparseArrays, DiffEqFlux, DifferentialEquations
using Flux: onehotbatch, onecold, logitcrossentropy, throttle
using Flux: @epochs
using Statistics: mean

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
train_X = Matrix{Float32}(features)  # dim: num_features * num_nodes
train_y = Float32.(labels)  # dim: target_catg * num_nodes
fg = FeaturedGraph(g)

# Define the Neural GDE
diffeqarray_to_array(x) = reshape(cpu(x), size(x)[1:2])

node = NeuralODE(
    GCNConv(fg, hidden=>hidden),
    (0.f0, 1.f0), Tsit5(), save_everystep = false,
    reltol = 1e-3, abstol = 1e-3, save_start = false
)

model = Chain(GCNConv(fg, num_features=>hidden, relu),
              Dropout(0.5),
              node,
              diffeqarray_to_array,
              GCNConv(fg, hidden=>target_catg),
              softmax)

# Loss
loss(x, y) = logitcrossentropy(model(x), y)
accuracy(x, y) = mean(onecold(model(x)) .== onecold(y))

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
