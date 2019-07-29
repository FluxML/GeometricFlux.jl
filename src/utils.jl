function neighboring(X::AbstractMatrix{T}, adjlist::AbstractVector{<:AbstractVector}) where {T}
    F, N = size(X)
    # TODO: Benchmark how many N or F need to switch from sequential to milti-thread version.
    ne_N = mapreduce(length, +, adjlist)
    Y = Matrix{T}(undef, F, ne_N)
    cluster = Vector{Int}(undef, ne_N)
    j = 0
    @inbounds for i = 1:N
        ne = adjlist[i]
        n = length(ne)
        Y[:, j+1:j+n] = view(X, :, ne)
        cluster[j+1:j+n] .= i
        j += n
    end
    Y, cluster
end

# function neighboring(X::AbstractMatrix{T}, adjlist::AbstractVector{<:AbstractVector}) where {T}
#     F, N = size(X)
#     starts = cumsum(map(length, adjlist))
#     ne_N = pop!(starts)
#     pushfirst!(starts, 0)
#     Y = Matrix{T}(undef, F, ne_N)
#     cluster = Vector{Int}(undef, ne_N)
#     @inbounds Threads.@threads for i = 1:N
#         ne = adjlist[i]
#         j = starts[i]
#         n = length(ne)
#         Y[:, j+1:j+n] = view(X, :, ne)
#         cluster[j+1:j+n] .= i
#     end
#     Y, cluster
# end
