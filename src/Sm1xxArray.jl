const k1, m2, m1 = 4, 4, 32

struct Sm1xxArray{T,N,X<:AbstractArray{T}} <: AbstractArray{T,N}
    ndims::Val{N}
    x::X
end

Base.parent(s::Sm1xxArray) = s.x

function Sm1xxArray(x::AbstractArray)
    ndims(x) >= 5 || throw(ArgumentError("Sm1xx array must have at least 5 dimensions"))
    size(x)[1:3] == (k1, m2, m1) || throw(ArgumentError("Size of Sm1xx array be ($k1, $m2, $m1, ...)"))
    N = ndims(x) - 3
    return Sm1xxArray(Val(N), x)
end

function Base.size(s::Sm1xxArray)
    @assert ndims(parent(s)) >= 5
    k1, m2, m1, k0, m0, rest... = size(parent(s))
    K = k1 * k0
    M  = m2 * m1 * m0
    return (K, M, rest...)
end

Base.IndexStyle(::Type{<:Sm1xxArray}) = IndexCartesian()
function Base.getindex(s::Sm1xxArray{T,N}, i::Vararg{Int,N}) where {T,N}
    @boundscheck checkbounds(s, i...)
    (; m1, m2, k1) = Sm1xx_sizes
    k, m, rest... = i

    k0i, k1i = fldmod1(k, k1)
    q,   m1i = fldmod1(m, m1)
    m0i, m2i = fldmod1(q, m2)

    return @inbounds parent(s)[k1i, m2i, m1i, k0i, m0i, rest...]
end

using Einops: @rearrange

function sm1xx(x::AbstractArray)
    ndims(x) >= 2 || throw(ArgumentError("Dense array must have at least 2 dimensions to be converted to Sm1xx"))
    size(x, 1) >= k1 && size(x, 2) >= m1 * m2 || throw(ArgumentError("Size of Dense array must be (>$k1, >$(m1*m2), ...)"))
    x′ = @rearrange(x, "(k1 k0) (m1 m2 m0) ... -> k1 m2 m1 k0 m0 ..."; k1, m2, m1)
    return Sm1xxArray(x′)
end

function Base.copy(s::Sm1xxArray)
    x = parent(s)
    return @rearrange(x, "k1 m2 m1 k0 m0 ... -> (k1 k0) (m1 m2 m0) ...")
end

### Broadcasting
Base.broadcastable(s::Sm1xxArray) = copy(s)

Base.print_array(io::IO, s::Sm1xxArray) = Base.print_array(io, Base.broadcastable(s))
