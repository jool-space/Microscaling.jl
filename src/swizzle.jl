using Adapt
using Einops
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

# The layout's type identity. Factor names are erased at construction — each
# factor becomes its position in the dense side's flat order — so any naming
# of the same layout produces the identical concrete type, and aliases like
# `F8_4x128Array` dispatch by plain equality. `L`/`R` are the canonicalized
# side trees (Int leaves), and `C` holds the declared context size per factor
# (`nothing` where the size is free): a layout declared with different fixed
# sizes is a different layout. Purely type-level; never instantiated.
struct Pattern{L,R,C} end

canonical(side, pos) = map(g -> g isa Symbol ? pos[g] : map(f -> pos[f], g), side)

"""
    SwizzledArray(storage, pattern; context...)
    SwizzledArray(storage, preset::Symbol)

Present array `storage`, laid out according to the right side of the Einops
`pattern`, as its dense equivalent described by the left side. `getindex`
performs the factor decomposition/composition implied by the pattern, and
`copy` rearranges the storage back into a dense array. Factor sizes are
taken from the storage dimensions, with the `context` supplying (and
checking) any that a grouped storage axis cannot determine alone.

The layout's type identity is its [`Pattern`](@ref): the grouping and
permutation with factor names canonicalized to positions, plus the sizes
declared in the `context`. Two spellings of the same layout produce the
identical concrete type — the names survive only as a runtime field (an
`NTuple{F,Symbol}`, whose type is name-free); `Einops.ArrowPattern(s)`
reconstructs the spelled pattern from them.

Use [`swizzle`](@ref) to produce one from a dense array.
"""
struct SwizzledArray{T,N,P<:Pattern,X<:AbstractArray{T},F} <: AbstractArray{T,N}
    x::X                     # swizzled storage, laid out per the pattern's right side
    dims::NTuple{N,Int}      # dense size, per the pattern's left side
    sizes::NTuple{F,Int}     # resolved factor sizes, in canonical position order
    names::NTuple{F,Symbol}  # the pattern's factor names (display only)
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
    sizesc = map(n -> sizes[n], names)
    dims = (map(g -> g isa Symbol ? sizes[g] : prod(f -> sizes[f], g), L)...,
            size(x)[length(R)+1:end]...)
    N = length(dims)
    ctx = NamedTuple(context)
    pos = NamedTuple{names}(ntuple(identity, length(names)))
    C = map(n -> haskey(ctx, n) ? Int(ctx[n]) : nothing, names)
    P = Pattern{canonical(L, pos),canonical(R, pos),C}
    return SwizzledArray{T,N,P,typeof(x),length(names)}(x, dims, sizesc, names)
end

"""
    Einops.ArrowPattern(s::SwizzledArray)

Reconstruct the (fixed part of the) Einops pattern `s` was swizzled with,
using the factor names it was constructed under.
"""
function Einops.ArrowPattern(s::SwizzledArray{T,N,Pattern{L,R,C}}) where {T,N,L,R,C}
    name(g) = g isa Int ? s.names[g] : map(f -> s.names[f], g)
    return map(name, L) --> map(name, R)
end

"""
    swizzle(x::AbstractArray, pattern; context...)
    swizzle(x::AbstractArray, preset::Symbol)

Rearrange dense array `x` into the swizzled storage layout described by the
Einops `pattern` (dense side --> storage side) and wrap the result in a
[`SwizzledArray`](@ref) presenting the original dense shape. A `preset`
symbol names a `pattern`/`context` pair from `Microscaling.SWIZZLES`:

  - `:f8_4x128` — the Blackwell block-scale-factor layout, which NVIDIA's
    docs call the "128×4 tiled layout" (cuDNN `F8_128x4`, CUTLASS Sm1xx;
    TransformerEngine likewise calls producing it "swizzling the scaling
    factors"):
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

# Dispatch alias for arrays swizzled with the :f8_4x128 layout. Names are
# canonicalized out of the type, so any naming of this grouping/permutation
# with the declared 4/4/32 tile sizes is this exact concrete type; a crossed
# mapping or different declared sizes is a different `Pattern`.
const F8_4x128Array{T,N,X} = SwizzledArray{T,N,
    Pattern{((1, 2), (3, 4, 5)),           # (k1 k0) (m1 m2 m0)
            (1, 4, 3, 2, 5),               # k1 m2 m1 k0 m0
            (4, nothing, 32, 4, nothing)}, # k1 = 4, m1 = 32, m2 = 4
    X, 5}
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
@generated function Base.getindex(s::SwizzledArray{T,N,Pattern{L,R,C}},
                                  I::Vararg{Int,N}) where {T,N,L,R,C}
    vals = Dict{Int,Any}()
    stmts = Expr[]
    for (axis, group) in enumerate(L)
        if group isa Int
            vals[group] = :(I[$axis])
        else
            prev = :(I[$axis])
            for j in 1:length(group)-1
                f = group[j]
                q, r = gensym(:q), gensym(:r)
                push!(stmts, :(($q, $r) = fldmod1($prev, s.sizes[$f])))
                vals[f] = r
                prev = q
            end
            vals[group[end]] = prev
        end
    end
    ix = Any[]
    for group in R
        if group isa Int
            push!(ix, vals[group])
        else
            ex = vals[group[end]]
            for j in length(group)-1:-1:1
                f = group[j]
                ex = :(($ex - 1) * s.sizes[$f] + $(vals[f]))
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

@generated function Base.copy(s::SwizzledArray{T,N,Pattern{L,R,C}}) where {T,N,L,R,C}
    name(i) = Symbol(:x, i)
    lift(side) = map(g -> g isa Int ? name(g) : map(name, g), side)
    Lp, Rp = lift(L), lift(R)
    p = N > length(L) ? (Rp..., ..) --> (Lp..., ..) : Rp --> Lp
    nf = sum(g -> g isa Int ? 1 : length(g), L)
    keys = ntuple(name, nf)
    return :(rearrange(parent(s), $p; NamedTuple{$keys}(s.sizes)...))
end

function Adapt.adapt_structure(to, s::SwizzledArray{T,N,P,X,F}) where {T,N,P,X,F}
    x = adapt(to, parent(s))
    return SwizzledArray{eltype(x),N,P,typeof(x),F}(x, s.dims, s.sizes, s.names)
end

### Display
#
# The summary elides the Pattern parameters — the swizzle string carries the
# same information readably — and appends the pattern plus its declared sizes.
sidestring(side, names) =
    join(map(g -> g isa Int ? String(names[g]) :
                  "(" * join((names[f] for f in g), " ") * ")", side), " ")

function Base.showarg(io::IO, s::SwizzledArray{T,N,Pattern{L,R,C},X,F},
                      toplevel) where {T,N,L,R,C,X,F}
    if isdefined(Base, :make_typealias) && Base.make_typealias(typeof(s)) !== nothing
        show(io, typeof(s))  # an alias (e.g. F8_4x128Array) prints compactly
    else
        print(io, "SwizzledArray{", T, ", ", N, ", Pattern{…}, ", X, ", ", F, "}")
    end
    toplevel || return
    print(" with swizzle")
    l, r = sidestring(L, s.names), sidestring(R, s.names)
    N > length(L) && (l *= " ..."; r *= " ...")
    print(io, " ")
    printstyled(io, "\"", l, " -> ", r, "\""; color=:green)
    declared = [i for i in 1:F if C[i] !== nothing]
    isempty(declared) || print(io, " where ")
    for (j, i) in enumerate(declared)
        j == 1 || print(io, ", ")
        print(io, s.names[i], "=")
        printstyled(io, C[i]; color=:cyan)
    end
    return
end

### Broadcasting
Base.broadcastable(s::SwizzledArray) = copy(s)

Base.print_array(io::IO, s::SwizzledArray) = Base.print_array(io, Base.broadcastable(s))
