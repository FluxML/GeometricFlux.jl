using CUDA
using Flux
using Flux: onecold
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

function load_data(dataset, batch_size, train_repeats=32, test_repeats=2)
    # (train_X, train_y) dim: (num_features, target_dim) × 2708
    train_X, train_y = map(x -> Matrix(x), alldata(Planetoid(), dataset, padding=true))
    # (test_X, test_y) dim: (num_features, target_dim) × 2708
    test_X, test_y = map(x -> Matrix(x), testdata(Planetoid(), dataset, padding=true))
    g = graphdata(Planetoid(), dataset)
    train_idx = 1:size(train_X, 2)
    test_idx = test_indices(Planetoid(), dataset)

    add_all_self_loops!(g)
    fg = FeaturedGraph(g)
    train_data = (repeat(train_X, outer=(1,1,train_repeats)), repeat(train_y, outer=(1,1,train_repeats)))
    test_data = (repeat(test_X, outer=(1,1,test_repeats)), repeat(test_y, outer=(1,1,test_repeats)))
    train_loader = DataLoader(train_data, batchsize=batch_size, shuffle=true)
    test_loader = DataLoader(test_data, batchsize=batch_size, shuffle=true)
    return train_loader, test_loader, fg, train_idx, test_idx
end

function add_all_self_loops!(g)
    for i in vertices(g)
        add_edge!(g, i, i)
    end
    return g
end

@with_kw mutable struct Args
    η = 0.01                # learning rate
    batch_size = 8          # batch size
    epochs = 20             # number of epochs
    seed = 0                # random seed
    cuda = true             # use GPU
    heads = 8               # attention heads
    input_dim = 1433        # input dimension
    hidden_dim = 16         # hidden dimension
    target_dim = 7          # target dimension
end

## Loss: cross entropy
model_loss(model, X, y, idx) =
    logitcrossentropy(model(X)[:,idx,:], y[:,idx,:])

accuracy(model, X::AbstractArray, y::AbstractArray, idx) =
    mean(onecold(softmax(cpu(model(X))[:,idx,:])) .== onecold(cpu(y)[:,idx,:]))

accuracy(model, loader::DataLoader, device, idx) =
    mean(accuracy(model, X |> device, y |> device, idx) for (X, y) in loader)

function train(; kws...)
    # load hyperparamters
    args = Args(; kws...)
    args.seed > 0 && Random.seed!(args.seed)

    # GPU config
    if args.cuda && CUDA.has_cuda()
        device = gpu
        @info "Training on GPU"
    else
        device = cpu
        @info "Training on CPU"
    end

    # load Cora from Planetoid dataset
    train_loader, test_loader, fg, train_idx, test_idx = load_data(:cora, args.batch_size)
    
    # build model
    model = Chain(
        WithGraph(fg, GATConv(args.input_dim=>args.hidden_dim, heads=args.heads)),
        Dropout(0.6),
        WithGraph(fg, GATConv(args.hidden_dim*args.heads=>args.target_dim, heads=args.heads, concat=false)),
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
            loss, back = Flux.pullback(ps) do
                model_loss(model, X |> device, y |> device, train_idx |> device)
            end
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
