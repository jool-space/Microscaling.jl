module Blockscaling

export BlockscalingFormat
export BlockscaledArray, BlockscaledVector, BlockscaledMatrix
public element_type, scale_type, block_size, block_dim

struct BlockscalingFormat{K,Scale<:Number,Element<:Number}
    block_size::Val{K}
    BlockscalingFormat(K::Int, S::Type, E::Type) = new{K, S, E}()
end

block_size(::BlockscalingFormat{K}) where K = K
scale_type(::BlockscalingFormat{K,S}) where {K,S} = S 
element_type(::BlockscalingFormat{K,S,E}) where {K,S,E} = E

struct BlockscaledArray{
    T<:Number, N, BD,
    K, Scale<:Number, Element<:Number,
    X<:AbstractArray{Scale,N}, P<:AbstractArray{Element,N},
} <: AbstractArray{T,N}
    eltype::Type{T}
    block_dim::Val{BD}
    format::BlockscalingFormat{K,Scale,Element}
    x::X
    p::P
end

function BlockscaledArray(
    T::Type, block_dim::Val,
    f::BlockscalingFormat,
    x::AbstractArray, p::AbstractArray
)
    scale_type(f) == eltype(x) || throw(ArgumentError("BlockscalingFormat scale type does not match scale array"))
    element_type(f) == eltype(p) || throw(ArgumentError("BlockscalingFormat element type does not match element array"))
    throw(MethodError(BlockscaledArray, (T, block_dim, f, x, p)))
end

function BlockscaledArray(
    T::Type, f::BlockscalingFormat, x::AbstractArray, p::AbstractArray;
    block_dim::Int=1
)
    expected_size = ntuple(Val(ndims(x))) do i
        i == block_dim ? size(x, i) * block_size(f) : size(x, i)
    end
    expected_size[block_dim] == size(p, block_dim) ||
        throw(ArgumentError("Expected block dimension ($block_dim) of the element array (size $(size(p, block_dim))) " *
            "to be $(block_size(f)) times greater than the corresponding dimension of the scale array (size $(size(x, block_dim)))"))
    expected_size == size(p) ||
        throw(ArgumentError("Expected non-block dimensions of the element array and scale array to match"))
    return BlockscaledArray(T, Val(block_dim), f, x, p)
end

BlockscaledArray{T}(args...; kws...) where T = BlockscaledArray(T, args...; kws...)

function BlockscaledArray(f::BlockscalingFormat, args...; kws...)
    T = promote_type(scale_type(f), element_type(f))
    isabstracttype(T) && (T = Float32)
    BlockscaledArray(T, f, args...; kws...)
end

Base.size(arr::BlockscaledArray, args...) = size(arr.p, args...)
block_dim(::BlockscaledArray{N,BD}) where {N,BD} = BD
block_size(arr::BlockscaledArray) = block_size(arr.format)
scale_type(arr::BlockscaledArray) = scale_type(arr.format)
element_type(arr::BlockscaledArray) = element_type(arr.format)

Base.IndexStyle(::Type{<:BlockscaledArray}) = IndexCartesian()

function Base.getindex(arr::BlockscaledArray{T,N,BD}, i::Vararg{Int,N}) where {T,N,BD}
    iₚ = i
    iₓ = ntuple(Val(N)) do j
        j == BD ? fld1(i[j], block_size(arr)) : i[j]
    end
    element = arr.p[iₚ...]
    scale = arr.x[iₓ...]
    value = T(element) * T(scale)
    return value
end

const BlockscaledVector{T} = BlockscaledArray{T,1}
const BlockscaledMatrix{T} = BlockscaledArray{T,2}


### Broadcasting

using Einops: @rearrange

function Base.broadcastable(arr::BlockscaledArray{T,N,1}) where {T,N}
    x_singleton = @rearrange(Base.broadcastable(arr.x), "k ... -> 1 k ...")
    p_block     = @rearrange(Base.broadcastable(arr.p), "(b k) ... -> b k ... "; b=block_size(arr))
    v = @rearrange(T.(x_singleton) .* T.(p_block), "b k ... -> (b k) ...")
    return v
end

Base.print_array(io::IO, arr::BlockscaledArray) = Base.print_array(io, Base.broadcastable(arr))

end
