using CUDA
using Flux
using Flux: onecold
using Flux.Losses: mae
using Flux.Data: DataLoader
using GeometricFlux
using Graphs
using GraphSignals
using MLDatasets
using Parameters: @with_kw
using ProgressMeter: Progress, next!
using Statistics
using Random

function load_data(
    batch_size,
    num_train_examples,
    num_test_examples,
    train_max_length,
    test_min_length,
    test_max_length
)
    train_data, test_data = MNIST(:train), MNIST(:test)
    train_X, train_y = shuffle_data(train_data.features, train_data.targets)
    test_X, test_y = shuffle_data(test_data.features, test_data.targets)

    train_data = generate_featuredgraphs(train_X, train_y, num_train_examples, 1:train_max_length)
    test_data = generate_featuredgraphs(test_X, test_y, num_test_examples, test_min_length:test_max_length)

    train_loader = DataLoader(train_data, batchsize=batch_size)
    test_loader = DataLoader(test_data, batchsize=batch_size)
    return train_loader, test_loader
end

function shuffle_data(X, y)
    X = reshape(X, :, size(y)...)
    p = randperm(size(y)...)
    return X[:,p], y[p]
end

function generate_featuredgraphs(X, y, num_examples, len_range)
    len = size(y, 1)
    data = []
    start = 1
    for _ in 1:num_examples
        n = rand(len_range)
        if start+n-1 > len
            start = 1
        end
        last = start + n - 1
        g = SimpleGraph(n)
        d = (FeaturedGraph(g, nf=X[:,start:last]), sum(y[start:last], dims=1))
        push!(data, d)
        start = last + 1
    end
    return data
end

@with_kw mutable struct Args
    η = 1e-4                     # learning rate
    num_train_examples = 1.5e5   # number of training examples
    num_test_examples = 1e4      # number of testing examples
    train_max_length = 10        # max number of digits in a training example
    test_min_length = 5          # min number of digits in a testing example
    test_max_length = 55         # max number of digits in a testing example
    batch_size = 128             # batch size
    epochs = 10                  # number of epochs
    seed = 0                     # random seed
    cuda = true                  # use GPU
    input_dim = 28*28            # input dimension
    hidden_dims = [300, 100, 30] # hidden dimension
    target_dim = 1               # target dimension
end

function model_loss(model, batch)
    ŷ = vcat(map(x -> global_feature(model(x[1])), batch)...)
    y = vcat(map(x -> x[2], batch)...)
    return mae(ŷ, y)
end

model_loss(model, loader::DataLoader, device) = mean(model_loss(model, batch |> device) for batch in loader)

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

    # load MNIST dataset
    train_loader, test_loader = load_data(
        args.batch_size,
        args.num_train_examples,
        args.num_test_examples,
        args.train_max_length,
        args.test_min_length,
        args.test_max_length
    )
    
    # build model
    ϕ = Chain(
        Dense(args.input_dim, args.hidden_dims[1], tanh),
        Dense(args.hidden_dims[1], args.hidden_dims[2], tanh),
        Dense(args.hidden_dims[2], args.hidden_dims[3], tanh),
    )
    ρ = Dense(args.hidden_dims[3], args.target_dim)
    model = DeepSet(ϕ, ρ) |> device

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

        for batch in train_loader
            batch = batch |> device
            train_loss, back = Flux.pullback(() -> model_loss(model, batch), ps)
            test_loss = model_loss(model, test_loader, device)
            grad = back(1f0)
            Flux.Optimise.update!(opt, ps, grad)

            # progress meter
            next!(progress; showvalues=[
                (:train_loss, train_loss),
                (:test_loss, test_loss)
            ])

            train_steps += 1
        end
    end

    return model, args
end

model, args = train()
