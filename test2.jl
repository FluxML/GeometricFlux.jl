using GraphMLDatasets
using GeometricFlux
using Flux
using Flux: @functor, throttle

data = rawdata(Cora()) 
adj_mat = data[:graph] |> Matrix{Int}
X = data[:all_X] |> Matrix{Float32}
y = data[:all_y] |> Vector{Int}

struct GNN
    conv1
    conv2 
    dense

    function GNN()
        new(GCNConv(1024=>512, relu),
            GCNConv(512=>128, relu), 
            Dense(128, 10))
    end
end

@functor GNN

function (net::GNN)(g, x)
    x = net.conv1(g, x)
    x = dropout(x, 0.5)
    x = net.conv2(g, x)
    x = net.dense(x)
    return x
end

net = GNN()

# model = Chain(GCNConv(g, 1024=>512, relu),
#               Dropout(0.5),
#               GCNConv(g, 512=>128),
#               Dense(128, 10),
#               softmax)
## Loss
loss(g, x, y) = logitcrossentropy(model(g, x), y)
accuracy(g, x, y) = mean(onecold(model(g, x)) .== onecold(y))

## Training
ps = Flux.params(model)
train_data = [(adj_mat, X, y)]
opt = ADAM(0.01)
evalcb() = @show(accuracy(adj_mat, X, y))

@show loss(first(train_data)...)

Flux.train!(loss, ps, train_data, opt, cb=throttle(evalcb, 10))