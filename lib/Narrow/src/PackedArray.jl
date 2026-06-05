struct PackedArray{T,N,E<:NarrowArray{T},A<:DenseArray{E,N}} <: AbstractArray{T,N}
    parent::A
end

pack_count(::Type{T}) where T = 8 ÷ gcd(bitwidth(T), 8)
function PackedArray{T}(arr::AbstractArray{T}) where T
    return PackedArray(NarrowVector.(reinterpret(NTuple{pack_count(T),T}, arr)))
end

const PackedVector{T} = PackedArray{T,1}
const PackedMatrix{T} = PackedArray{T,2}

Base.parent(arr::PackedArray) = arr.parent

inner_size(arr::PackedArray, i::Int) = size(eltype(parent(arr)), i)
inner_size(arr::PackedArray) = ntuple(i -> inner_size(arr, i), Val(ndims(arr)))

Base.size(arr::PackedArray, i::Int) = size(parent(arr), i) * size(eltype(parent(arr)), i)
Base.size(arr::PackedArray) = ntuple(i -> size(arr, i), Val(ndims(arr)))

Base.IndexStyle(::Type{<:PackedArray}) = IndexCartesian()
function Base.getindex(arr::PackedArray{T,N}, i::Vararg{Int,N}) where {T,N}
    outer_inner_i = ntuple(j -> fldmod1(i[j], inner_size(arr, j)), Val(N))
    outer_i = first.(outer_inner_i)
    inner_i = last.(outer_inner_i)
    return parent(arr)[outer_i...][inner_i...]
end

function Base.reinterpret(::Type{T}, arr::PackedArray) where T
    return reinterpret(T, parent(arr))
end

function Broadcast.broadcastable(arr::PackedArray{T,<:Any,<:NarrowVector}) where T
    return reinterpret(T, map(SArray, parent(arr)))
end

Base.print_array(io::IO, arr::PackedArray) = Base.print_array(io, Base.broadcastable(arr))
