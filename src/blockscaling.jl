using Adapt
using Rewrap: Unsqueeze, Keep, Split, Merge

"""
    BlockScaledArray{T}(scale, element, block_size = size(element) .÷ size(scale))
    BlockScaledArray(scale, element, args...)

An `AbstractArray{T,N}` pairing an `element` array with a `scale` array,
where each scale applies to one block of elements. Indexing returns
`T(element) * T(scale)`.

`block_size` gives the block extent along each dimension: an `Int` `k` means
`k` consecutive elements share a scale, and `:` means the whole dimension
shares a scale. Without `T`, the element type is promoted from
`eltype(scale)` and `eltype(element)` (`Float32` if the promotion is
abstract).
"""
struct BlockScaledArray{
    T<:Number, N, K<:NTuple{N,Any},
    S<:AbstractArray{<:Number,N}, E<:AbstractArray{<:Number,N},
} <: AbstractArray{T,N}
    scale::S
    element::E
end

function BlockScaledArray{T,N,K}(scale::S, element::E) where {
    T<:Number, N, K<:NTuple{N,Any},
    S<:AbstractArray{<:Number,N}, E<:AbstractArray{<:Number,N}
}
    validate_block_size(Tuple(K.parameters), size(scale), size(element))
    return BlockScaledArray{T,N,K,S,E}(scale, element)
end

function validate_block_size(block_size, scale_size, element_size)
    for (i, (k, xs, ps)) in enumerate(zip(block_size, scale_size, element_size))
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

function BlockScaledArray{T}(
    scale::AbstractArray{<:Number,N},
    element::AbstractArray{<:Number,N},
    block_size::NTuple{N,Union{Int,Colon}} = size(element) .÷ size(scale)
) where {T,N}
    K = Tuple{block_size...}
    return BlockScaledArray{T,N,K}(scale, element)
end

function promote_eltype(scale::AbstractArray, element::AbstractArray)
    T = promote_type(eltype(scale), eltype(element))
    isabstracttype(T) ? Float32 : T
end

function BlockScaledArray(scale::AbstractArray, element::AbstractArray, args...; kws...)
    return BlockScaledArray{promote_eltype(scale, element)}(scale, element, args...; kws...)
end

"""
    elements(arr::BlockScaledArray)

The element array, in which each element belongs to a block sharing one scale.
"""
elements(arr::BlockScaledArray) = arr.element

"""
    scales(arr::BlockScaledArray)

The scale array, holding one scale per block of elements.
"""
scales(arr::BlockScaledArray) = arr.scale

Base.size(arr::BlockScaledArray, args...) = size(elements(arr), args...)

"""
    block_size(arr::BlockScaledArray[, i])

The block extent along each dimension, as a tuple of `Int`s and `Colon`s —
or just its `i`-th entry.
"""
block_size(::BlockScaledArray{T,N,K}) where {T,N,K} = Tuple(K.parameters)
block_size(arr::BlockScaledArray, i::Integer) = block_size(arr)[i]

"""
    scale_type(arr::BlockScaledArray)

The element type of the scale array.
"""
scale_type(arr::BlockScaledArray) = eltype(scales(arr))

"""
    element_type(arr::BlockScaledArray)

The element type of the element array.
"""
element_type(arr::BlockScaledArray) = eltype(elements(arr))

Base.IndexStyle(::Type{<:BlockScaledArray}) = IndexCartesian()

function Adapt.adapt_structure(
    to, arr::BlockScaledArray{T,N,K}
) where {T,N,K}
    scale = adapt(to, scales(arr))
    element = adapt(to, elements(arr))
    return BlockScaledArray{T,N,K}(scale, element)
end

# Allocate plain (dense) output in the SAME storage as the element data, so
# `similar(arr, T, dims)` on a GPU-backed BlockScaledArray yields a GPU array
# rather than Base's default CPU `Array`. Reductions/maps rely on this for the
# destination buffer.
Base.similar(arr::BlockScaledArray, ::Type{T}, dims::Dims) where {T} =
    similar(elements(arr), T, dims)

function Base.getindex(arr::BlockScaledArray{T,N}, i::Vararg{Int,N}) where {T,N}
    @boundscheck checkbounds(arr, i...)
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
    element = elements(arr)[iₚ...]
    scale = scales(arr)[iₓ...]
    value = T(element) * T(scale)
    return value
end

"""
    BlockScaledVector{T}

Alias for [`BlockScaledArray`](@ref)`{T,1}`.
"""
const BlockScaledVector{T} = BlockScaledArray{T,1}

"""
    BlockScaledMatrix{T}

Alias for [`BlockScaledArray`](@ref)`{T,2}`.
"""
const BlockScaledMatrix{T} = BlockScaledArray{T,2}

function Base.copy(arr::BlockScaledArray{T,N}) where {T,N}
    scale_singleton = reshape(T.(scales(arr)), ntuple(i -> isodd(i) ? Unsqueeze() : Keep(), Val(2N)))
    element_shape = ntuple(Val(N)) do i
        k = block_size(arr, i)
        k = k isa Colon ? size(arr, i) : k
        Split(1, (k, :))
    end
    element_block = reshape(T.(elements(arr)), element_shape)
    v = scale_singleton .* element_block
    return reshape(v, ntuple(Returns(Merge(2)), Val(N)))
end

### Broadcasting

Base.broadcastable(arr::BlockScaledArray) = copy(arr)

Base.print_array(io::IO, arr::BlockScaledArray) = Base.print_array(io, copy(arr))
