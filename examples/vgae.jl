using CUDA
using Flux
using Flux: onecold
using Flux.Losses: logitbinarycrossentropy
using Flux.Data: DataLoader
using GeometricFlux
using GeometricFlux.Datasets
using Graphs
using GraphSignals
using Parameters: @with_kw
using ProgressMeter: Progress, next!
using Statistics
using Random

function load_data(dataset, batch_size, train_repeats=4)
    s, t = dataset[1].edge_index
    g = Graphs.Graph(dataset[1].num_nodes)
    for (i, j) in zip(s, t)
        Graphs.add_edge!(g, i, j)
    end
    fg = FeaturedGraph(g)
    A = GraphSignals.adjacency_matrix(fg)

    data = dataset[1].node_data
    X = data.features

    train_X = repeat(X, outer=(1, 1, train_repeats))
    train_y = repeat(A, outer=(1, 1, train_repeats))
    loader = DataLoader((train_X, train_y), batchsize=batch_size, shuffle=true)
    return loader, fg
end

@with_kw mutable struct Args
    η = 0.01                # learning rate
    β = 1.0                 # information bottleneck paramater
    batch_size = 1          # batch size
    epochs = 200            # number of epochs
    seed = 0                # random seed
    cuda = true             # use GPU
    input_dim = 1433        # input dimension
    h_dim = 32              # hidden dimension
    z_dim = 16              # latent dimension
    dataset = Cora          # dataset to train on
end

## Loss

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

function precision(model, X::AbstractArray, A::AbstractArray)
    ŷ = cpu(Flux.flatten(model(X))) .≥ 0.5
    y = cpu(Flux.flatten(A))
    return sum(y .* ŷ) / sum(ŷ)
end

precision(model, loader::DataLoader, device) =
    mean(precision(model, X |> device, A |> device) for (X, A) in loader)

function train(; kws...)
    # load hyperparamters
    args = Args(; kws...)
    args.seed > 0 && Random.seed!(args.seed)

    # GPU config
    if args.cuda && CUDA.has_cuda()
        device = gpu
        CUDA.allowscalar(false)
        @info "Training on GPU"
    else
        device = cpu
        @info "Training on CPU"
    end

    # load Cora from Planetoid dataset
    loader, fg = load_data(args.dataset(), args.batch_size)
    
    # build model
    encoder = VariationalGraphEncoder(
        WithGraph(fg, GCNConv(args.input_dim=>args.h_dim, relu)),
        WithGraph(fg, GCNConv(args.h_dim=>args.z_dim)),
        WithGraph(fg, GCNConv(args.h_dim=>args.z_dim)),
        args.z_dim
    )

    decoder = InnerProductDecoder(σ)
    
    model = VGAE(encoder, decoder) |> device

    # Adam optimizer
    opt = Adam(args.η)
    
    # parameters
    ps = Flux.params(model)

    # training
    train_steps = 0
    @info "Start Training, total $(args.epochs) epochs"
    for epoch = 1:args.epochs
        @info "Epoch $(epoch)"
        progress = Progress(length(loader))

        for (X, Â) in loader
            X, Â = X |> device, Â |> device
            loss, back = Flux.pullback(() -> model_loss(model, X, Â, args.β), ps)
            prec = precision(model, loader, device)
            grad = back(1f0)
            Flux.Optimise.update!(opt, ps, grad)

            # progress meter
            next!(progress; showvalues=[
                (:loss, loss),
                (:precision, prec),
            ])

            train_steps += 1
        end
    end

    return model, args
end

model, args = train()
