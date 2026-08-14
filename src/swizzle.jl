using Adapt
using Einops               # rearrange, -->, and the .. ellipsis (not importable by name)
using Einops: ArrowPattern

# ## Pattern plumbing
#
# A swizzle is described by an Einops `ArrowPattern`: the left side is the
# dense (logical) layout, the right side the swizzled storage layout. Both
# sides are tuples of factor symbols or grouped factor tuples (column-major:
# first factor fastest), optionally ending in a shared trailing ellipsis for
# batch dimensions that pass through untouched.

function fixed_side(side::Tuple)
    hasrest = !isempty(side) && side[end] isa typeof(..)
    fixed = hasrest ? Base.front(side) : side
    for e in fixed
        e isa Symbol || (e isa Tuple && !isempty(e) && all(f -> f isa Symbol, e)) ||
            throw(ArgumentError("swizzle patterns support factor symbols, tuples \
                                 of factor symbols, and one trailing ellipsis; got $e"))
    end
    return fixed, hasrest
end

function factors(side)
    fs = Symbol[]
    for e in side
        e isa Symbol ? push!(fs, e) : append!(fs, e)
    end
    return fs
end

function sides(p::ArrowPattern{L,R}) where {L,R}
    l, lrest = fixed_side(L)
    r, rrest = fixed_side(R)
    lrest == rrest ||
        throw(ArgumentError("ellipsis must appear on both sides or neither: $p"))
    lf, rf = factors(l), factors(r)
    allunique(lf) && allunique(rf) && Set(lf) == Set(rf) ||
        throw(ArgumentError("pattern sides must use the same factors, once each: $p"))
    return l, r, lrest
end

"""
    SwizzledArray(storage, pattern; context...)
    SwizzledArray(storage, preset::Symbol)

Present array `storage`, laid out according to the right side of the Einops
`pattern`, as its dense equivalent described by the left side. `getindex`
performs the factor decomposition/composition implied by the pattern, and
`copy` rearranges the storage back into a dense array. Factor sizes are
taken from the storage dimensions, with the `context` supplying (and
checking) any that a grouped storage axis cannot determine alone.

Use [`swizzle`](@ref) to produce one from a dense array.
"""
struct SwizzledArray{T,N,L,R,X<:AbstractArray{T},S<:NamedTuple} <: AbstractArray{T,N}
    x::X                 # swizzled storage, laid out per the pattern's right side
    dims::NTuple{N,Int}  # dense size, per the pattern's left side
    sizes::S             # resolved size of every factor in the pattern
end

Base.parent(s::SwizzledArray) = s.x
Base.size(s::SwizzledArray) = s.dims

function SwizzledArray(x::AbstractArray{T}, p::ArrowPattern; context...) where T
    L, R, rest = sides(p)
    (rest ? ndims(x) >= length(R) : ndims(x) == length(R)) ||
        throw(ArgumentError("storage of rank $(ndims(x)) does not match the \
                             $(length(R))-axis storage side of $p"))
    sizes = Dict{Symbol,Int}(pairs(NamedTuple(context)))
    for (axis, group) in enumerate(R)
        sz = size(x, axis)
        if group isa Symbol
            get!(sizes, group, sz) == sz || throw(DimensionMismatch(
                "storage axis $axis has size $sz, but $group = $(sizes[group])"))
        else
            unknown = [f for f in group if !haskey(sizes, f)]
            known = prod(Int[sizes[f] for f in group if haskey(sizes, f)]; init=1)
            if isempty(unknown)
                known == sz || throw(DimensionMismatch(
                    "storage axis $axis has size $sz, but $group = $known"))
            elseif length(unknown) == 1
                sz % known == 0 || throw(DimensionMismatch(
                    "storage axis $axis of size $sz is not divisible by $known \
                     to determine $(only(unknown))"))
                sizes[only(unknown)] = sz ÷ known
            else
                throw(ArgumentError("cannot determine sizes of $(Tuple(unknown)) \
                                     from storage axis $axis; pass them in the context"))
            end
        end
    end
    names = Tuple(factors(L))
    sizesnt = NamedTuple{names}(map(n -> sizes[n], names))
    dims = (map(g -> g isa Symbol ? sizes[g] : prod(f -> sizes[f], g), L)...,
            size(x)[length(R)+1:end]...)
    N = length(dims)
    return SwizzledArray{T,N,L,R,typeof(x),typeof(sizesnt)}(x, dims, sizesnt)
end

"""
    swizzle(x::AbstractArray, pattern; context...)
    swizzle(x::AbstractArray, preset::Symbol)

Rearrange dense array `x` into the swizzled storage layout described by the
Einops `pattern` (dense side --> storage side) and wrap the result in a
[`SwizzledArray`](@ref) presenting the original dense shape. A `preset`
symbol names a `pattern`/`context` pair from `Microscaling.SWIZZLES`:

  - `:f8_4x128` — the Blackwell block-scale-factor layout
    (cuDNN `F8_128x4`, CUTLASS Sm1xx):
    `((:k1, :k0), (:m1, :m2, :m0), ..) --> (:k1, :m2, :m1, :k0, :m0, ..)`
    with `k1 = 4, m2 = 4, m1 = 32`. Each tile serves 4 scale columns by 128
    rows, the `(4, 4, 32)` leading storage dims being the tile-internal
    warp-group interleave.
"""
function swizzle(x::AbstractArray, p::ArrowPattern; context...)
    ctx = NamedTuple(context)
    L, _, rest = sides(p)
    (rest ? ndims(x) >= length(L) : ndims(x) == length(L)) ||
        throw(ArgumentError("array of rank $(ndims(x)) does not match the \
                             $(length(L))-axis dense side of $p"))
    for (axis, group) in enumerate(L)
        group isa Symbol && continue
        known = prod(Int[ctx[f] for f in group if haskey(ctx, f)]; init=1)
        size(x, axis) % known == 0 || throw(ArgumentError(
            "axis $axis of size $(size(x, axis)) is not divisible by $known \
             as $group requires"))
    end
    return SwizzledArray(rearrange(x, p; ctx...), p; ctx...)
end

const SWIZZLES = (;
    f8_4x128 = (; pattern = ((:k1, :k0), (:m1, :m2, :m0), ..) -->
                            (:k1, :m2, :m1, :k0, :m0, ..),
                  context = (; k1 = 4, m2 = 4, m1 = 32)),
)

preset(name::Symbol) = haskey(SWIZZLES, name) ? SWIZZLES[name] :
    throw(ArgumentError("unknown swizzle preset :$name; available: \
                         $(join(map(k -> ":$k", keys(SWIZZLES)), ", "))"))

swizzle(x::AbstractArray, name::Symbol) =
    (p = preset(name); swizzle(x, p.pattern; p.context...))
SwizzledArray(x::AbstractArray, name::Symbol) =
    (p = preset(name); SwizzledArray(x, p.pattern; p.context...))

# dispatch alias for arrays swizzled with the :f8_4x128 preset
const F8_4x128Array{T,N,X,S} = SwizzledArray{T,N,
    ((:k1, :k0), (:m1, :m2, :m0)), (:k1, :m2, :m1, :k0, :m0), X, S}
F8_4x128Array(x::AbstractArray) = SwizzledArray(x, :f8_4x128)

"""
    f8_4x128(x::AbstractArray)

Shorthand for [`swizzle`](@ref)`(x, :f8_4x128)`.
"""
f8_4x128(x::AbstractArray) = swizzle(x, :f8_4x128)

# ## Element access
#
# Generated from the pattern: decompose each dense index over its left-side
# group (first factor fastest), then compose the factor indices back per the
# right-side storage groups. Trailing ellipsis dims pass through untouched.

Base.IndexStyle(::Type{<:SwizzledArray}) = IndexCartesian()
@generated function Base.getindex(s::SwizzledArray{T,N,L,R}, I::Vararg{Int,N}) where {T,N,L,R}
    vals = Dict{Symbol,Any}()
    stmts = Expr[]
    for (axis, group) in enumerate(L)
        if group isa Symbol
            vals[group] = :(I[$axis])
        else
            prev = :(I[$axis])
            for j in 1:length(group)-1
                f = group[j]
                q, r = gensym(:q), gensym(f)
                push!(stmts, :(($q, $r) = fldmod1($prev, s.sizes.$f)))
                vals[f] = r
                prev = q
            end
            vals[group[end]] = prev
        end
    end
    ix = Any[]
    for group in R
        if group isa Symbol
            push!(ix, vals[group])
        else
            ex = vals[group[end]]
            for j in length(group)-1:-1:1
                f = group[j]
                ex = :(($ex - 1) * s.sizes.$f + $(vals[f]))
            end
            push!(ix, ex)
        end
    end
    append!(ix, (:(I[$k]) for k in length(L)+1:N))
    return quote
        @boundscheck checkbounds(s, I...)
        $(stmts...)
        return @inbounds parent(s)[$(ix...)]
    end
end

@generated function Base.copy(s::SwizzledArray{T,N,L,R}) where {T,N,L,R}
    p = N > length(L) ? (R..., ..) --> (L..., ..) : R --> L
    return :(rearrange(parent(s), $p; s.sizes...))
end

function Adapt.adapt_structure(to, s::SwizzledArray{T,N,L,R}) where {T,N,L,R}
    x = adapt(to, parent(s))
    return SwizzledArray{eltype(x),N,L,R,typeof(x),typeof(s.sizes)}(x, s.dims, s.sizes)
end

### Broadcasting
Base.broadcastable(s::SwizzledArray) = copy(s)

Base.print_array(io::IO, s::SwizzledArray) = Base.print_array(io, Base.broadcastable(s))
