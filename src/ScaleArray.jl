abstract type ScaleLayout end

struct ScaleArray{L<:ScaleLayout,N,T,X<:AbstractArray{T}} <: AbstractArray{T,N}
    layout::L
    ndims::Val{N}
    x::X
end

ScaleLayout(s::ScaleArray) = s.layout
Base.parent(s::ScaleArray) = s.x

### Dense

struct Dense <: ScaleLayout end

ScaleArray(layout::Dense, x::AbstractArray) = ScaleArray(layout, Val(ndims(x)), x)
Base.size(s::ScaleArray{Dense}, args...) = size(parent(s), args...)
Base.IndexStyle(::Type{<:ScaleArray{Dense}}) = IndexLinear()
Base.getindex(s::ScaleArray{Dense}, i::Int) = parent(s)[i]

# Dense is the default layout
ScaleArray(x::AbstractArray) = ScaleArray(Dense(), x)

### Sm1xx

struct Sm1xx <: ScaleLayout end
sizes(::Sm1xx) = (; k1 = 4, m2 = 4, m1 = 32)

function ScaleArray(layout::Sm1xx, x::AbstractArray)
    ndims(x) >= 5 || throw(ArgumentError("Sm1xx array must have at least 5 dimensions"))
    (; m1, m2, k1) = sizes(layout)
    size(x)[1:3] == (k1, m2, m1) || throw(ArgumentError("Size of Sm1xx array be ($k1, $m2, $m1, ...)"))
    N = ndims(x) - 3
    return ScaleArray(layout, Val(N), x)
end

function Base.size(s::ScaleArray{Sm1xx})
    @assert ndims(parent(s)) >= 5
    k1, m2, m1, k0, m0, rest... = size(parent(s))
    K = k1 * k0
    M  = m2 * m1 * m0
    return (K, M, rest...)
end

Base.IndexStyle(::Type{<:ScaleArray}) = IndexCartesian()
function Base.getindex(s::ScaleArray{Sm1xx,N,T}, i::Vararg{Int,N}) where {T,N}
    @boundscheck checkbounds(s, i...)
    (; m1, m2, k1) = sizes(ScaleLayout(s))
    k, m, rest... = i

    k0i, k1i = fldmod1(k, k1)
    q,   m1i = fldmod1(m, m1)
    m0i, m2i = fldmod1(q, m2)

    return @inbounds parent(s)[k1i, m2i, m1i, k0i, m0i, rest...]
end

### Conversion

relayout(::L, s::ScaleArray{L}) where L<:ScaleLayout = s
relayout(layout::ScaleLayout, s::ScaleArray) = throw(MethodError(relayout, (layout, s)))
relayout(layout::ScaleLayout, s::AbstractArray) = relayout(layout, ScaleArray(s))

using Einops: @rearrange

function relayout(layout::Sm1xx, s::ScaleArray{Dense})
    ndims(s) >= 2 || throw(ArgumentError("Dense array must have at least 2 dimensions to be converted to Sm1xx"))
    size(s, 1) >= k1 && size(s, 2) >= m1 * m2 || throw(ArgumentError("Size of Dense array must be (>$k1, >$(m1*m2), ...)"))
    x = parent(s)
    x′ = @rearrange(x, "(k1 k0) (m1 m2 m0) ... -> k1 m2 m1 k0 m0 ..."; sizes(layout)...)
    s′ = ScaleArray(layout, x′)
    return s′
end

function relayout(layout::Dense, s::ScaleArray{Sm1xx})
    x = parent(s)
    x′ = @rearrange(x, "k1 m2 m1 k0 m0 ... -> (k1 k0) (m1 m2 m0) ..."; sizes(layout)...)
    s′ = ScaleArray(layout, x′)
    return s′
end

### Broadcasting

Base.broadcastable(s::ScaleArray{Dense}) = Base.broadcastable(parent(s))

Base.broadcastable(s::ScaleArray{Sm1xx}) = Base.broadcastable(relayout(Dense(), s))
