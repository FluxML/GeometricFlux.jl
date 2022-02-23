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

function load_data(dataset, batch_size, train_repeats=512, test_repeats=32)
    # (train_X, train_y) dim: (num_features, target_dim) × 2708
    train_X, train_y = map(x -> Matrix(x), alldata(Planetoid(), dataset, padding=true))
    # (test_X, test_y) dim: (num_features, target_dim) × 2708
    test_X, test_y = map(x -> Matrix(x), testdata(Planetoid(), dataset, padding=true))
    g = graphdata(Planetoid(), dataset)
    train_idx = 1:size(train_X, 2)
    test_idx = test_indices(Planetoid(), dataset)

    fg = FeaturedGraph(g)
    train_data = (repeat(train_X, outer=(1,1,train_repeats)), repeat(train_y, outer=(1,1,train_repeats)))
    test_data = (repeat(test_X, outer=(1,1,test_repeats)), repeat(test_y, outer=(1,1,test_repeats)))
    train_loader = DataLoader(train_data, batchsize=batch_size, shuffle=true)
    test_loader = DataLoader(test_data, batchsize=batch_size, shuffle=true)
    return train_loader, test_loader, fg, train_idx, test_idx
end

@with_kw mutable struct Args
    η = 0.01                # learning rate
    batch_size = 64         # batch size
    epochs = 200            # number of epochs
    seed = 0                # random seed
    cuda = true             # use GPU
    input_dim = 1433        # input dimension
    hidden_dim = 16         # hidden dimension
    target_dim = 7          # target dimension
end

## Loss: cross entropy
function model_loss(model, X, y, idx)
    return logitcrossentropy(model(X)[:,idx,:], y[:,idx,:])
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
        WithGraph(fg, GraphConv(fg, args.input_dim=>args.hidden_dim, relu)),
        Dropout(0.5),
        WithGraph(fg, GraphConv(fg, args.hidden_dim=>args.target_dim)),
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

# model, args = train()

T = Float32
in_channel = 3
in_channel_edge = 1
out_channel = 5

N = 4
E = 4
adj = T[0. 1. 0. 1.;
        1. 0. 1. 0.;
        0. 1. 0. 1.;
        1. 0. 1. 0.]
fg = FeaturedGraph(adj)

batch_size = 10
X = rand(T, in_channel, N, batch_size)
gc = GraphConv(fg, in_channel=>out_channel)
Y = gc(X)
g = Zygote.gradient(() -> sum(gc(X)), Flux.params(gc))
