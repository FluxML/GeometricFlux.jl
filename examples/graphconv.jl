using CUDA
using Flux
using Flux: onehotbatch, onecold
using Flux.Losses: logitcrossentropy
using Flux.Data: DataLoader
using GeometricFlux
using GeometricFlux.Datasets
using GraphSignals
using Graphs
using Parameters: @with_kw
using ProgressMeter: Progress, next!
using Statistics
using Random

function load_data(dataset, batch_size, train_repeats=64, test_repeats=2)
    s, t = dataset[1].edge_index
    g = Graphs.Graph(dataset[1].num_nodes)
    for (i, j) in zip(s, t)
        Graphs.add_edge!(g, i, j)
    end

    data = dataset[1].node_data
    X, y = data.features, onehotbatch(data.targets, 1:7)
    train_idx, test_idx = data.train_mask, data.val_mask

    # (train_X, train_y) dim: (num_features, target_dim) × 2708 × train_repeats
    train_X, train_y = repeat(X, outer=(1,1,train_repeats)), repeat(y, outer=(1,1,train_repeats))
    # (test_X, test_y) dim: (num_features, target_dim) × 2708 × test_repeats
    test_X, test_y = repeat(X, outer=(1,1,test_repeats)), repeat(y, outer=(1,1,test_repeats))

    fg = FeaturedGraph(g)
    train_loader = DataLoader((train_X, train_y), batchsize=batch_size, shuffle=true)
    test_loader = DataLoader((test_X, test_y), batchsize=batch_size, shuffle=true)
    return train_loader, test_loader, fg, train_idx, test_idx
end

@with_kw mutable struct Args
    η = 0.01                # learning rate
    batch_size = 16         # batch size
    epochs = 200            # number of epochs
    seed = 0                # random seed
    cuda = true             # use GPU
    input_dim = 1433        # input dimension
    hidden_dim = 16         # hidden dimension
    target_dim = 7          # target dimension
    dataset = Cora          # dataset to train on
end

## Loss: cross entropy
model_loss(model, X, y, idx) = logitcrossentropy(model(X)[:,idx,:], y[:,idx,:])

accuracy(model, X::AbstractArray, y::AbstractArray, idx) =
    mean(onecold(softmax(cpu(model(X))[:,idx,:])) .== onecold(cpu(y)[:,idx,:]))

accuracy(model, loader::DataLoader, device, idx) = mean(accuracy(model, X |> device, y |> device, idx) for (X, y) in loader)

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
    train_loader, test_loader, fg, train_idx, test_idx = load_data(args.dataset(), args.batch_size)
    
    # build model
    model = Chain(
        WithGraph(fg, GraphConv(args.input_dim=>args.hidden_dim, relu)),
        WithGraph(fg, GraphConv(args.hidden_dim=>args.target_dim)),
    ) |> device

    # Adam optimizer
    opt = Adam(args.η)
    
    # parameters
    ps = Flux.params(model)

    # training
    train_steps = 0
    @info "Start Training, total $(args.epochs) epochs"
    for epoch = 1:args.epochs
        @info "Epoch $(epoch)"
        progress = Progress(length(train_loader))

        for (X, y) in train_loader
            X, y, device_idx = X |> device, y |> device, train_idx |> device
            loss, back = Flux.pullback(() -> model_loss(model, X, y, device_idx), ps)
            train_acc = accuracy(model, train_loader, device, train_idx)
            test_acc = accuracy(model, test_loader, device, test_idx)
            grad = back(1f0)
            Flux.Optimise.update!(opt, ps, grad)

            # progress meter
            next!(progress; showvalues=[
                (:loss, loss),
                (:train_accuracy, train_acc),
                (:test_accuracy, test_acc)
            ])

            train_steps += 1
        end
    end

    return model, args
end

model, args = train()
