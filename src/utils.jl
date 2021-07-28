

function check_num_nodes(fg::FeaturedGraph, x::AbstractArray)
    @assert nv(fg) == size(x, ndims(x))    
end