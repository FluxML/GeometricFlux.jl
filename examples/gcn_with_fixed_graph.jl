using CUDA
using Flux
using Flux: onehotbatch, onecold
using Flux.Losses: logitcrossentropy
using Flux.Data: DataLoader
using GeometricFlux
using GeometricFlux.Datasets
using GraphSignals
using Logging: with_logger
using Parameters: @with_kw
using ProgressMeter: Progress, next!
using Statistics
using Random

function load_data(dataset, batch_size)
    # (train_X, train_y) dim: (num_features, target_dim) × 1708
    train_X, train_y = map(x -> Matrix(x), alldata(Planetoid(), dataset))
    # (test_X, test_y) dim: (num_features, target_dim) × 1000
    test_X, test_y = map(x -> Matrix(x), testdata(Planetoid(), dataset))
    g = graphdata(Planetoid(), dataset)
    train_idx = 1:size(train_X, 2)
    test_idx = test_indices(Planetoid(), dataset)

    # padding zeros
    tr_X = zeros(Float32, size(train_X, 1), size(train_X, 2) + size(test_X, 2))
    te_X = zeros(Float32, size(test_X, 1), size(train_X, 2) + size(test_X, 2))
    tr_y = zeros(Float32, size(train_y, 1), size(train_y, 2) + size(test_y, 2))
    te_y = zeros(Float32, size(test_y, 1), size(train_y, 2) + size(test_y, 2))
    tr_X[:, train_idx] .= train_X
    te_X[:, test_idx] .= test_X
    tr_y[:, train_idx] .= train_y
    te_y[:, test_idx] .= test_y

    fg = FeaturedGraph(g)
    train_data = (repeat(tr_X, outer=(1,1,256)), repeat(tr_y, outer=(1,1,256)))
    test_data = (repeat(te_X, outer=(1,1,32)), repeat(te_y, outer=(1,1,32)))
    train_loader = DataLoader(train_data, batchsize=batch_size, shuffle=true)
    test_loader = DataLoader(test_data, batchsize=batch_size, shuffle=true)
    return train_loader, test_loader, fg, train_idx, test_idx
end

@with_kw mutable struct Args
    η = 0.01                # learning rate
    λ = 5f-4                # regularization paramater
    batch_size = 32         # batch size
    num_nodes = 2708        # number of nodes for graph
    epochs = 200            # number of epochs
    seed = 0                # random seed
    cuda = true             # use GPU
    input_dim = 1433        # input dimension
    hidden_dim = 16         # hidden dimension
    target_dim = 7          # target dimension
end

## Loss: cross entropy with first layer L2 regularization 
l2norm(x) = sum(abs2, x)
function model_loss(model, λ, X, y, idx)
    loss = logitcrossentropy(model(X)[:,idx,:], y[:,idx,:])
    loss += λ*sum(l2norm, Flux.params(model[1]))
    return loss
end

function accuracy(model, X::AbstractArray, y::AbstractArray, idx)
    return mean(onecold(softmax(cpu(model(X))[:,idx,:])) .== onecold(cpu(y)[:,idx,:]))
end

accuracy(model, loader::DataLoader, device, idx) = mean(accuracy(model, X |> device, y |> device, idx) for (X, y) in loader)

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
        WithGraph(fg, GCNConv(args.input_dim=>args.hidden_dim, relu)),
        Dropout(0.5),
        WithGraph(fg, GCNConv(args.hidden_dim=>args.target_dim)),
    ) |> device

    # ADAM optimizer
    opt = ADAM(args.η)
    
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
                model_loss(model, args.λ, X |> device, y |> device, train_idx |> device)
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
