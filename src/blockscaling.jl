module Blockscaling

export BlockscaledArray, BlockscaledVector, BlockscaledMatrix
public block_size, scale_type, element_type

struct BlockscaledArray{
    T<:Number, N, K<:Tuple,
    X<:AbstractArray{<:Number,N}, P<:AbstractArray{<:Number,N},
} <: AbstractArray{T,N}
    x::X
    p::P
end

function BlockscaledArray{T,N,K}(
    x::X,
    p::P
) where {
    T<:Number, N, K<:Tuple,
    X<:AbstractArray{<:Number,N}, P<:AbstractArray{<:Number,N},
}
    return BlockscaledArray{T,N,K,X,P}(x, p)
end

function BlockscaledArray{T}(
    block_size::Dims{N},
    x::AbstractArray{<:Number,N},
    p::AbstractArray{<:Number,N}
) where {T,N}
    K = Tuple{block_size...}
    return BlockscaledArray{T,N,K}(x, p)
end

function BlockscaledArray{T}(
    x::AbstractArray{<:Number,N},
    p::AbstractArray{<:Number,N}
) where {T,N}
    block_size = ntuple(i -> size(p, i) ÷ size(x, i), Val(N))
    return BlockscaledArray{T}(block_size, x, p)
end

function BlockscaledArray(x::AbstractArray, p::AbstractArray)
    T = promote_type(eltype(x), eltype(p))
    isabstracttype(T) && (T = Float32)
    return BlockscaledArray{T}(x, p)
end

Base.size(arr::BlockscaledArray, args...) = size(arr.p, args...)
block_size(::BlockscaledArray{T,N,K}) where {T,N,K} = Tuple(K.parameters)
block_size(arr::BlockscaledArray, i::Integer) = block_size(arr)[i]
scale_type(arr::BlockscaledArray) = eltype(arr.x)
element_type(arr::BlockscaledArray) = eltype(arr.p)

Base.IndexStyle(::Type{<:BlockscaledArray}) = IndexCartesian()

function Base.getindex(arr::BlockscaledArray{T,N}, i::Vararg{Int,N}) where {T,N}
    iₚ = i
    iₓ = ntuple(Val(N)) do j
        k = block_size(arr, j)
        isone(k) ? i[j] : fld1(i[j], k)
    end
    element = arr.p[iₚ...]
    scale = arr.x[iₓ...]
    value = T(element) * T(scale)
    return value
end

const BlockscaledVector{T} = BlockscaledArray{T,1}
const BlockscaledMatrix{T} = BlockscaledArray{T,2}

using Rewrap

function Base.copy(arr::BlockscaledArray{T,N}) where {T,N}
    x_singleton = reshape(copy(arr.x), ntuple(i -> isodd(i) ? Unsqueeze() : Keep(), Val(2N)))
    p_block     = reshape(copy(arr.p), ntuple(i -> Split(1, (block_size(arr, i), :)), Val(N)))
    v = T.(x_singleton) .* T.(p_block)
    return reshape(v, ntuple(Returns(Merge(2)), Val(N)))
end

### Broadcasting

Base.broadcastable(arr::BlockscaledArray) = copy(arr)

Base.print_array(io::IO, arr::BlockscaledArray) = Base.print_array(io, Base.broadcastable(arr))

end
