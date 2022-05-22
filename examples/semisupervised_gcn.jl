using CUDA
using Flux
using Flux: onecold
using Flux.Losses: logitcrossentropy
using Flux.Data: DataLoader
using GeometricFlux
using GeometricFlux.Datasets
using GraphSignals
using Parameters: @with_kw
using ProgressMeter: Progress, next!
using Statistics
using Random

function load_data(dataset, batch_size, train_repeats=256, test_repeats=32)
    # (train_X, train_y) dim: (num_features, target_dim) × 140
    train_X, train_y = map(x->Matrix(x), traindata(Planetoid(), dataset))
    # (test_X, test_y) dim: (num_features, target_dim) × 1000
    test_X, test_y = map(x->Matrix(x), testdata(Planetoid(), dataset))
    g = graphdata(Planetoid(), dataset)
    train_idx = train_indices(Planetoid(), dataset)
    test_idx = test_indices(Planetoid(), dataset)

    train_data = [(subgraph(FeaturedGraph(g, nf=train_X), train_idx), train_y) for _ in 1:train_repeats]
    test_data = [(subgraph(FeaturedGraph(g, nf=test_X), test_idx), test_y) for _ in 1:test_repeats]
    train_batch = Flux.batch(train_data)
    test_batch = Flux.batch(test_data)

    train_loader = DataLoader(train_batch, batchsize=batch_size, shuffle=true)
    test_loader = DataLoader(test_batch, batchsize=batch_size, shuffle=true)
    return train_loader, test_loader
end

@with_kw mutable struct Args
    η = 0.01                # learning rate
    λ = 5f-4                # regularization paramater
    batch_size = 32         # batch size
    epochs = 200            # number of epochs
    seed = 0                # random seed
    cuda = true             # use GPU
    input_dim = 1433        # input dimension
    hidden_dim = 16         # hidden dimension
    target_dim = 7          # target dimension
end

## Loss: cross entropy with first layer L2 regularization
l2norm(x) = sum(abs2, x)

function model_loss(model, λ, batch, batch_size::Int)
    loss = 0.f0
    for (x, y) in [[batch[1][i], batch[2][:,:,i]] for i = 1:batch_size]
        loss += logitcrossentropy(model(x), y)
        loss += λ*sum(l2norm, Flux.params(model[1]))
    end
    return loss
end

function accuracy(model, batch::Tuple{AbstractVector, AbstractArray}, batch_size::Int)
    return mean(mean(onecold(softmax(cpu(model(x)))) .== onecold(cpu(y))) for (x,y) in [[batch[1][i], batch[2][:,:,i]] for i = 1:batch_size])
end

accuracy(model, loader::DataLoader, device, batch_size::Int) = mean(accuracy(model, batch |> device, batch_size) for batch in loader)

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
    train_loader, test_loader = load_data(:cora, args.batch_size)

    # build model
    model = Chain(
        GCNConv(args.input_dim=>args.hidden_dim, relu),
        GraphParallel(node_layer=Dropout(0.5)),
        GCNConv(args.hidden_dim=>args.target_dim),
        node_feature,
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
        for batch in train_loader
            loss, back = Flux.pullback(ps) do
                model_loss(model, args.λ, batch |> device, args.batch_size)
            end
            train_acc = accuracy(model, train_loader, device, args.batch_size)
            test_acc = accuracy(model, test_loader, device, args.batch_size)
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
