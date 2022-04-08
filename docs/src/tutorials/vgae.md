# Variational Graph Autoencoder

Variational Graph Autoencoder (VGAE) is a unsupervised generative model. It takes node features and graph structure and predicts the edge link in the graph. A link preidction task is defined for this model.

## Step 1: Load Dataset

We load dataset from Planetoid dataset. Here cora dataset is used.

```julia
train_X, _ = map(x -> Matrix(x), alldata(Planetoid(), dataset))
```

Notably, a link prediction task will output a graph in the form of adjacency matrix, so an adjacency matrix is needed as label for this task.

```julia
g = graphdata(Planetoid(), dataset)
fg = FeaturedGraph(g)
A = GraphSignals.adjacency_matrix(fg)
```

## Step 2: Batch up Features and Labels

Just batch up features as usual.

```julia
data = (repeat(X, outer=(1,1,train_repeats)), repeat(A, outer=(1,1,train_repeats)))
loader = DataLoader(data, batchsize=batch_size, shuffle=true)
```

## Step 3: Build a VGAE model

A VGAE model is composed of an encoder and a decoder. A `VariationalGraphEncoder` is used as an graph encoder and it contains a neural network to encode node features. A `InnerProductDecoder` is the decoder to predict links in a graph. Actually, it gives an adjacency matrix. Finally, we build `VGAE` model with encoder and decoder.

```julia
encoder = VariationalGraphEncoder(
    WithGraph(fg, GCNConv(args.input_dim=>args.h_dim, relu)),
    WithGraph(fg, GCNConv(args.h_dim=>args.z_dim)),
    WithGraph(fg, GCNConv(args.h_dim=>args.z_dim)),
    args.z_dim
)

decoder = InnerProductDecoder(σ)

model = VGAE(encoder, decoder) |> device
```

## Step 4: Loss Functions and Link Prediction

Since a VGAE is a VAE model, its loss function is composed of a KL divergence and a log P.

```julia
function kldivergence(model, X::AbstractArray{T}) where {T}
    μ̂, logσ̂ = GeometricFlux.summarize(model.encoder, X)
    return -T(0.5) * sum(one(T) .+ T(2).*logσ̂ .- μ̂.^2 .- exp.(T(2).*logσ̂))
end

function logp(model, X, Y)
    Z = model.encoder(X)
    return -logitbinarycrossentropy(model.decoder(Z), Y)
end

function model_loss(model, X, Y, β)
    kl_q_p = kldivergence(model, X)
    logp_y_z = logp(model, X, Y)
    return -logp_y_z + β*kl_q_p
end
```

Precision metric is used to measure the existence of edges to be predicted from a model.

```julia
function precision(model, X::AbstractArray, A::AbstractArray)
    ŷ = cpu(Flux.flatten(model(X))) .≥ 0.5
    y = cpu(Flux.flatten(A))
    return sum(y .* ŷ) / sum(ŷ)
end
```

## Step 5: Training VGAE Model

```julia
# ADAM optimizer
opt = ADAM(args.η)

# parameters
ps = Flux.params(model)

# training
@info "Start Training, total $(args.epochs) epochs"
for epoch = 1:args.epochs
    @info "Epoch $(epoch)"

    for (X, A) in loader
        loss, back = Flux.pullback(ps) do
            model_loss(model, X |> device, A |> device, args.β)
        end
        prec = precision(model, loader, device)
        grad = back(1f0)
        Flux.Optimise.update!(opt, ps, grad)
    end
end
```

For a complete example, please check [examples/vgae.jl](../../examples/vgae.jl).
