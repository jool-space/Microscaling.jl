module Blockscaling

export BlockscaledArray, BlockscaledVector, BlockscaledMatrix
public block_size, scale_type, element_type

struct BlockscaledArray{
    T<:Number, N, K<:NTuple{N,Any},
    X<:AbstractArray{<:Number,N}, P<:AbstractArray{<:Number,N},
} <: AbstractArray{T,N}
    x::X
    p::P
end

function BlockscaledArray{T,N,K}(x::X, p::P) where {
    T<:Number, N, K<:NTuple{N,Any},
    X<:AbstractArray{<:Number,N}, P<:AbstractArray{<:Number,N}
}
    return BlockscaledArray{T,N,K,X,P}(x, p)
end

function validate_block_size(block_size, x_size, p_size)
    for (i, (k, xs, ps)) in enumerate(zip(block_size, x_size, p_size))
        if k === 1
            xs == ps || throw(DimensionMismatch(
                "Expected number of scale values ($xs) and element values ($ps) to match along dimension $i, for block size of $k."))
        elseif k isa Colon
            isone(xs) || throw(DimensionMismatch(
                "Expected only one dimension-wide scale value along dimension $i, got $xs."))
        else
            @assert k isa Int
            k * xs == ps || throw(DimensionMismatch(
                "Expected number of scale values ($xs) times block size along dimension $i ($k) to be equal to the number of element values ($ps)."))
        end
    end
end

function BlockscaledArray{T}(
    x::AbstractArray{<:Number,N},
    p::AbstractArray{<:Number,N},
    block_size::NTuple{N,Union{Int,Colon}} = ntuple(i -> size(p, i) ÷ size(x, i), Val(N))
) where {T,N}
    validate_block_size(block_size, size(x), size(p))
    K = Tuple{block_size...}
    return BlockscaledArray{T,N,K}(x, p)
end

function BlockscaledArray(x::AbstractArray, p::AbstractArray, args...; kws...)
    T = promote_type(eltype(x), eltype(p))
    isabstracttype(T) && (T = Float32)
    return BlockscaledArray{T}(x, p, args...; kws...)
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
        if k === 1
            i[j]
        elseif k isa Colon
            1
        else
            fld1(i[j], k)
        end
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
    p_shape = ntuple(Val(N)) do i
        k = block_size(arr, i)
        Split(1, k isa Colon ? (:, 1) : (k, :))
    end
    p_block     = reshape(copy(arr.p), p_shape)
    v = T.(x_singleton) .* T.(p_block)
    return reshape(v, ntuple(Returns(Merge(2)), Val(N)))
end

### Broadcasting

Base.broadcastable(arr::BlockscaledArray) = copy(arr)

Base.print_array(io::IO, arr::BlockscaledArray) = Base.print_array(io, Base.broadcastable(arr))

end
