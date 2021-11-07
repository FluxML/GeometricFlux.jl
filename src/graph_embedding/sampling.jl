"""
Alias Sampling first described in [1]. [2] might be a helpful resource to understand alias sampling.

[1] A. Kronmal and A. V. Peterson. On the alias method for generating random variables from a discrete distribution. The American Statistician, 33(4):214-218, 1979.
[2] https://lips.cs.princeton.edu/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
"""

alias_setup(probs::Vector{Float64}) = alias_setup(sparse(probs))

"""
Computes alias probabilities.
"""
function alias_setup(probs::SparseVector{Float64})::Tuple{SparseVector{Int}, SparseVector{Float64}}
    K = length(probs)
    J = spzeros(Int, K)
    q = probs * K

    smaller::Vector{Int} = [] # prob idxs < 1/K
    larger::Vector{Int} = []  # prob idxs >= 1/k

    for i in 1:length(probs)
        if q[i] < 1.0  # equivalent to prob < 1/K but saves the division
            push!(smaller, i)
        else
            push!(larger, i)
        end
    end
    while length(smaller) > 0 && length(larger) > 0
        small = pop!(smaller)
        large = pop!(larger)
        J[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0
            push!(smaller, large)
        else
            push!(larger, large)
        end
    end

    return J, q
end

function alias_sample(J::SparseVector{Int}, q::SparseVector{Float64})::Int

    small_index = rand() * length(J) |> ceil |> Int
    if rand() < q[small_index]
        return small_index
    else
        return J[small_index]
    end
end
