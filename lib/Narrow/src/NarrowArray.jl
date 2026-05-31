using StaticArrays

function unsigned_type(N::Int)
    N ==  1 ? UInt8   :
    N ==  2 ? UInt16  :
    N ==  4 ? UInt32  :
    N ==  8 ? UInt64  :
    N == 16 ? UInt128 :
    error("Could not find a $N-byte Unsigned type")
end

const Bytes{N} = NTuple{N,UInt8}

struct NarrowArray{T,N,S<:NTuple{N,Any},D<:Bytes} <: StaticArray{S,T,N}
    data::D
    function NarrowArray{T,N,S}(data::D) where {T,N,S<:NTuple{N,Any},D<:Bytes}
        arr = new{T,N,S,D}(data)
        bitwidth(data) - 8 < bitwidth(arr) <= bitwidth(data) ||
            throw(ArgumentError("Expected $(cld(bitwidth(arr), 8)) byte(s) for $(length(arr)) $(bitwidth(T))-bit element(s), but data has $(length(data))."))
        return arr
    end
end

bitwidth(::Type{T}) where T<:NarrowArray = bitwidth(eltype(T)) * length(T)

function NarrowArray{T,N,S}(xs::NTuple{L,T}) where {T,N,S<:NTuple{N,Any},L}
    unpacked_bytes = reinterpret.(unsigned_type(sizeof(T)), xs)
    data = pack(Val(bitwidth(T)), unpacked_bytes)
    arr = NarrowArray{T,N,S}(data)
    length(arr) == L || throw(ArgumentError("Type expects $(length(arr)) element(s), but got $L."))
    return arr
end

function StaticArrays.similar_type(
    ::Type{<:NarrowArray{T,<:Any,S}},
    ::Type{T′}=T,
    ::Size{S′}=Size(S)
) where {T,S,T′,S′}
    return NarrowArray{T′,length(S′),Tuple{S′...}}
end

const NarrowVector{T,L} = NarrowArray{T,1,Tuple{L}}
const NarrowMatrix{T,S₁,S₂} = NarrowArray{T,2,Tuple{S₁,S₂}}

NarrowVector(xs::NTuple{L,T}) where {L,T} = NarrowVector{T,L}(xs)

narrow(xs...) = NarrowVector(xs)

function Base.Tuple(v::NarrowArray)
    unpacked_bytes = unpack(Val(bitwidth(eltype(v))), Val(length(v)), v.data)
    xs = reinterpret.(eltype(v), unpacked_bytes)::NTuple{length(v)}
    return xs
end

NarrowArray(xs::StaticArray{S,T}) where {N,S<:NTuple{N,Any},T} = NarrowArray{T,N,S}(Tuple(xs))

Base.IndexStyle(::Type{<:NarrowArray}) = IndexLinear()

function Base.getindex(arr::NarrowArray{T}, i::Int) where T
    W = bitwidth(T)
    S = length(arr)
    u = packed_getindex(Val(W), Val(S), arr.data, i)
    return reinterpret(T, u)
end

Base.broadcastable(v::NarrowArray{<:Any,<:Any,S}) where S = SArray{S}(Tuple(v))

Base.reinterpret(::Type{T}, v::NarrowArray) where T = reinterpret(T, v.data)
Base.reinterpret(::Type{Unsigned}, v::NarrowArray) = reinterpret(unsigned_type(sizeof(v)), v)
