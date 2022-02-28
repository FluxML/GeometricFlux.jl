# Message passing scheme

Message passing scheme is a popular GNN scheme in many frameworks. It adapts the property of connectivity of neighbors and form a general approach for spatial graph convolutional neural network. It comprises two user-defined functions and one aggregate function. A message function is defined to process information from edge states and node states from neighbors and itself. Messages from each node are obtained and aggregated by aggregate function to provide node-level information for update function. Update function takes current node state and aggregated message and gives a new node state.

Message passing scheme is realized into a abstract type `MessagePassing`. Any subtype of `MessagePassing` is a message passing layer which utilize default message and update functions:

```
message(mp, x_i, x_j, e_ij) = x_j
update(mp, m, x) = m
```

`mp` denotes a message passing layer. `message` accepts node state `x_i` for node `i` and its neighbor state `x_j` for node `j`, as well as corresponding edge state `e_ij` for edge `(i,j)`. The default message function gives all the neighbor state `x_j` for neighbor of node `i`. `update` takes aggregated message `m` and current node state `x`, and then outputs `m`.

```@docs
GeometricFlux.MessagePassing
```

## Message function

A message function accepts feature vector representing node state `x_i`, feature vectors for neighbor state `x_j` and corresponding edge state `e_ij`. A vector is expected to output from `message` for message. User can override `message` for customized message passing layer to provide desired behavior.

```@docs
GeometricFlux.message
```

## Aggregate messages

Messages from message function are aggregated by an aggregate function. An aggregated message is passed to update function for node-level computation. An aggregate function is given by the following:

```
propagate(mp, fg::FeaturedGraph, aggr::Symbol=:add)
```

`propagate` function calls the whole message passing layer. `fg` acts as an input for message passing layer and `aggr` represents assignment of aggregate function to `propagate` function. `:add` represents an aggregate function of addition of all messages.

The following `aggr` are available aggregate functions:

`:add`: sum over all messages
`:sub`: negative of sum over all messages
`:mul`: multiplication over all messages
`:div`: inverse of multiplication over all messages
`:max`: the maximum of all messages
`:min`: the minimum of all messages
`:mean`: the average of all messages

## Update function

An update function takes aggregated message `m` and current node state `x` as arguments. An output vector is expected to be the new node state for next layer. User can override `update` for customized message passing layer to provide desired behavior.

```@docs
GeometricFlux.update
```
