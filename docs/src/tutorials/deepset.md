# Predicting Digits Sum from DeepSet model

Digits sum is a task of summing up digits in images or text. This example demonstrates summing up digits in arbitrary number of MNIST images. To accomplish such task, DeepSet model is suitable for this task. DeepSet model is excellent at the task which takes a set of objects and reduces them into single object.

## Step 1: Load MNIST Dataset

Since a DeepSet model predicts the summation from a set of images, we have to prepare training dataset composed of a random-sized set of images and a summed result.

First, the whole dataset is loaded from MLDatasets.jl and then shuffled before generating training dataset.

```julia
train_X, train_y = MLDatasets.MNIST.traindata(Float32)
train_X, train_y = shuffle_data(train_X, train_y)
```

The `generate_featuredgraphs` here generates a set of pairs which contains a `FeaturedGraph` and a summed number for prediction target. In a `FeaturedGraph`, an arbitrary number of MNIST images are collected as node features and corresponding nodes are collected in a graph without edges.

```julia
train_data = generate_featuredgraphs(train_X, train_y, num_train_examples, 1:train_max_length)
```

`num_train_examples` is the parameter for assigning how many training example to generate. `1:train_max_length` specifies the range of number of images to contained in one example.

## Step 2: Build a DeepSet model

A DeepSet takes a set of objects and outputs single object. To make a model accept a set of objects, the model input must be invariant to permutation. The DeepSet model is simply composed of two parts: ``\phi`` network and ``\rho`` network. 

```math
Z = \rho ( \sum_{x_i \in \mathcal{V}} \phi (x_i) )
```

``\phi`` network embeds every images and they are summed up to be a single embedding. Permutation invariance comes from the use of summation. In general, a commutative binary operator can be used to reduce a set of embeddings into one embedding. Finally, ``\rho`` network decodes the embedding to a number.

```julia
ϕ = Chain(
    Dense(args.input_dim, args.hidden_dims[1], tanh),
    Dense(args.hidden_dims[1], args.hidden_dims[2], tanh),
    Dense(args.hidden_dims[2], args.hidden_dims[3], tanh),
)
ρ = Dense(args.hidden_dims[3], args.target_dim)
model = DeepSet(ϕ, ρ) |> device
```

## Step 3: Loss Functions

Mean absolute error is used as the loss function. Since the model outputs a `FeaturedGraph`, the prediction is placed as a global feature in `FeaturedGraph`.

```julia
function model_loss(model, batch)
    ŷ = vcat(map(x -> global_feature(model(x[1])), batch)...)
    y = vcat(map(x -> x[2], batch)...)
    return mae(ŷ, y)
end
```

## Step 4: Training DeepSet Model

```julia
# optimizer
opt = ADAM(args.η)

# parameters
ps = Flux.params(model)

# training
@info "Start Training, total $(args.epochs) epochs"
for epoch = 1:args.epochs
    @info "Epoch $(epoch)"

    for batch in train_loader
        train_loss, back = Flux.pullback(ps) do
            model_loss(model, batch |> device)
        end
        test_loss = model_loss(model, test_loader, device)
        grad = back(1f0)
        Flux.Optimise.update!(opt, ps, grad)
    end
end
```

For a complete example, please check [examples/digitsum_deepsets.jl](../../examples/digitsum_deepsets.jl).
