function hcat_by_sum(xs::AbstractVector)
    T = eltype(xs[1])
    dim = size(xs[1], 1)
    N = length(xs)

    ns = map(x->size(x,2), xs)
    pushfirst!(ns, 1)
    cumsum!(ns, ns)

    A = similar(xs[1], T, dim, ns[end]-1)
    for i in 1:N
        A[:, ns[i]:(ns[i+1]-1)] .= xs[i]
    end
    return A
end

function ChainRulesCore.rrule(::typeof(hcat_by_sum), xs::AbstractVector)
    N = length(xs)

    ns = map(x->size(x,2), xs)
    pushfirst!(ns, 1)
    cumsum!(ns, ns)

    hcat_by_sum_pullback(Δ) = (NoTangent(), ntuple(i->view(Δ,:,ns[i]:(ns[i+1]-1)), N))
    hcat_by_sum(xs), hcat_by_sum_pullback
end

function ChainRulesCore.rrule(::typeof(parent), A::Base.SubArray)
    parent_pullback(Δ) = (NoTangent(), view(Δ, A.indices...))
    parent(A), parent_pullback
end
