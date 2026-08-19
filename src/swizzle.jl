using Adapt
using Einops
using Einops: ArrowPattern

# ## Pattern plumbing
#
# A swizzle is an Einops `ArrowPattern`: dense layout left, storage layout
# right; groups are column-major, a shared trailing ellipsis passes batch
# dims through. Resolution is cold value code; the hot paths are generated.

function sides(p::ArrowPattern{L,R}) where {L,R}
    strip(side) = !isempty(side) && side[end] isa typeof(..) ?
                  (Base.front(side), true) : (side, false)
    (l, lrest), (r, rrest) = strip(L), strip(R)
    for e in (l..., r...)
        e isa Symbol || (e isa Tuple && !isempty(e) && all(f -> f isa Symbol, e)) ||
            throw(ArgumentError("swizzle patterns support factor symbols, tuples \
                                 of factor symbols, and one trailing ellipsis; got $e"))
    end
    lrest == rrest ||
        throw(ArgumentError("ellipsis must appear on both sides or neither: $p"))
    lf, rf = factors(l), factors(r)
    allunique(lf) && allunique(rf) && Set(lf) == Set(rf) ||
        throw(ArgumentError("pattern sides must use the same factors, once each: $p"))
    return l, r, lrest
end

factors(side) = Symbol[f for e in side for f in (e isa Symbol ? (e,) : e)]

# Type identity: the side trees with names canonicalized to dense-side
# positions, plus declared context sizes `C` (`nothing` = free) — any naming
# of one layout is one concrete type. Never instantiated.
struct Pattern{L,R,C} end

canonical(side, pos) = map(g -> g isa Symbol ? pos[g] : map(f -> pos[f], g), side)

# ## Resolution
#
# A name→size ledger (0 = unresolved), seeded from the context — whose names
# are checked, so a mistyped keyword errors — then solved group by group.

function ledger(names, ctx, p)
    unknown = Tuple(k for k in keys(ctx) if !(k in names))
    isempty(unknown) || throw(ArgumentError(
        "context names $unknown are not factors of $p"))
    sizes = zeros(Int, length(names))
    for (f, v) in pairs(ctx)
        sizes[findfirst(==(f), names)] = Int(v)
    end
    return sizes
end

function solve!(sizes, names, g, extent, axis)
    at(f) = findfirst(==(f), names)::Int
    if g isa Symbol
        i = at(g)
        sizes[i] == 0 ? (sizes[i] = extent) : sizes[i] == extent ||
            throw(DimensionMismatch(
                "axis $axis has extent $extent, but $g = $(sizes[i])"))
    else
        free = [f for f in g if sizes[at(f)] == 0]
        known = prod(Int[sizes[at(f)] for f in g if sizes[at(f)] != 0]; init=1)
        if isempty(free)
            known == extent || throw(DimensionMismatch(
                "axis $axis has extent $extent, but $g = $known"))
        elseif length(free) == 1
            extent % known == 0 || throw(DimensionMismatch(
                "axis $axis extent $extent is not divisible by $known \
                 to determine $(only(free))"))
            sizes[at(only(free))] = extent ÷ known
        else
            throw(ArgumentError("cannot determine sizes of $(Tuple(free)) \
                                 on axis $axis; pass them in the context"))
        end
    end
end

"""
    SwizzledArray(storage, pattern; context...)
    SwizzledArray(storage, left, right; context...)
    SwizzledArray(storage, preset::Symbol)

Present `storage`, laid out per the `pattern`'s right side, as the dense
array its left side describes. Factor sizes come from the storage dims;
`context` supplies (and checks) any a grouped axis cannot determine alone.

Type identity is the name-erased `Pattern` parameter, so two spellings of
one layout share a concrete type; `Einops.ArrowPattern(s)` respells it.
Use [`swizzle`](@ref) to produce one from a dense array.
"""
struct SwizzledArray{T,N,P<:Pattern,X<:AbstractArray{T},F} <: AbstractArray{T,N}
    x::X                     # swizzled storage (the pattern's right side)
    dims::NTuple{N,Int}      # dense size (the pattern's left side)
    sizes::NTuple{F,Int}     # factor sizes, in canonical position order
    names::NTuple{F,Symbol}  # factor names (display only)
end

Base.parent(s::SwizzledArray) = s.x
Base.size(s::SwizzledArray) = s.dims

function SwizzledArray(x::AbstractArray{T}, p::ArrowPattern; dims=nothing, context...) where T
    ctx = NamedTuple(context)
    L, R, rest = sides(p)
    (rest ? ndims(x) >= length(R) : ndims(x) == length(R)) ||
        throw(ArgumentError("storage of rank $(ndims(x)) does not match the \
                             $(length(R))-axis storage side of $p"))
    names = factors(L)
    sizes = ledger(names, ctx, p)
    for (axis, g) in enumerate(R)
        solve!(sizes, names, g, size(x, axis), axis)
    end
    fsize(f) = sizes[findfirst(==(f), names)]
    full = (map(g -> g isa Symbol ? fsize(g) : prod(fsize, g), L)...,
            size(x)[length(R)+1:end]...)
    # padded logical extents cannot be inferred from storage; take a kwarg
    if dims === nothing
        dims = full
    else
        length(dims) == length(L) || throw(ArgumentError(
            "logical dims cover the pattern's $(length(L)) dense axes, \
             got $(length(dims))"))
        all(map((d, f) -> 1 <= d <= f, Tuple(dims), full[1:length(L)])) ||
            throw(DimensionMismatch("logical dims $(Tuple(dims)) do not fit the \
                                     storage's dense extents $(full[1:length(L)])"))
        dims = (Int.(Tuple(dims))..., full[length(L)+1:end]...)
    end
    pos = NamedTuple{Tuple(names)}(ntuple(identity, length(names)))
    C = Tuple(haskey(ctx, f) ? Int(ctx[f]) : nothing for f in names)
    return SwizzledArray{T,length(dims),Pattern{canonical(L, pos),canonical(R, pos),C},
                         typeof(x),length(names)}(x, dims, Tuple(sizes), Tuple(names))
end

"""
    padded_size(s::SwizzledArray)

The physical dense extents implied by the factor sizes; `size(s)` when
unpadded.
"""
@generated function padded_size(s::SwizzledArray{T,N,Pattern{L,R,C}}) where {T,N,L,R,C}
    ex = Any[g isa Int ? :(s.sizes[$g]) : :(*($(map(f -> :(s.sizes[$f]), g)...)))
             for g in L]
    append!(ex, (:(size(parent(s), $(length(R) + k))) for k in 1:N-length(L)))
    return :(tuple($(ex...)))
end

"""
    Einops.ArrowPattern(s::SwizzledArray)

Respell the (fixed part of the) pattern `s` was swizzled with, using its
construction names.
"""
function Einops.ArrowPattern(s::SwizzledArray{T,N,Pattern{L,R,C}}) where {T,N,L,R,C}
    name(g) = g isa Int ? s.names[g] : map(f -> s.names[f], g)
    return map(name, L) --> map(name, R)
end

"""
    swizzle(x::AbstractArray, pattern; pad=true, context...)
    swizzle(x::AbstractArray, left, right; context...)
    swizzle(x::AbstractArray, preset::Symbol)

Rearrange dense `x` into the swizzled layout the Einops `pattern` describes
(dense side --> storage side) and wrap it in a [`SwizzledArray`](@ref)
presenting the original shape. A `preset` names an entry of
`Microscaling.SWIZZLES`; `:f8_4x128` is the Blackwell block-scale-factor
layout (NVIDIA's "128×4 tiled layout", cuDNN's `F8_128x4`, CUTLASS's Sm1xx):
4 scale columns × 128 rows per tile, interleaved as `(4, 4, 32)` leading
storage dims.

Axes that do not tile exactly are zero-filled to whole tiles — the vendor
padding contract. The wrapper keeps the logical shape, [`padded_size`](@ref)
gives the physical extents, `copy` slices the pad back off, and `pad=false`
refuses. Data movement delegates to [`swizzle!`](@ref).
"""
function swizzle(x::AbstractArray, p::ArrowPattern; pad::Bool=true, context...)
    ctx = NamedTuple(context)
    L, R, rest = sides(p)
    (rest ? ndims(x) >= length(L) : ndims(x) == length(L)) ||
        throw(ArgumentError("array of rank $(ndims(x)) does not match the \
                             $(length(L))-axis dense side of $p"))
    names = factors(L)
    sizes = ledger(names, ctx, p)
    np = length(L)
    targets = ntuple(np) do axis
        g = L[axis]
        sz = size(x, axis)
        t = if g isa Symbol
            get(ctx, g, sz)
        else
            known = prod(Int[ctx[f] for f in g if haskey(ctx, f)]; init=1)
            any(f -> !haskey(ctx, f), g) ? cld(sz, known) * known : known
        end
        t >= sz || throw(ArgumentError(
            "axis $axis of size $sz exceeds its declared extent $t"))
        t
    end
    logical = ntuple(axis -> size(x, axis), np)
    targets == logical || pad || throw(ArgumentError(
        "axes $logical do not tile the pattern's extents $targets exactly, \
         and `pad=false` refuses zero-fill padding"))
    for (axis, g) in enumerate(L)
        solve!(sizes, names, g, targets[axis], axis)
    end
    fsize(f) = sizes[findfirst(==(f), names)]
    rdims = map(g -> g isa Symbol ? fsize(g) : prod(fsize, g), R)
    storage = similar(x, (rdims..., size(x)[np+1:end]...))
    return swizzle!(SwizzledArray(storage, p; dims=logical, ctx...), x)
end

"""
    swizzle!(dest::SwizzledArray, x::AbstractArray)

Write dense `x` into `dest`'s swizzled storage in place: one `permutedims!`
between free reshapes (the storage-side factor order *is* the permutation)
when unpadded — allocation-free, safe under CUDA graph capture — or via a
transient zero-filled staging buffer, `Pol.release!`d after, when padded.
"""
function swizzle!(s::SwizzledArray{T,N}, x::AbstractArray) where {T,N}
    size(x) == size(s) || throw(DimensionMismatch(
        "cannot swizzle a $(size(x)) array into a $(size(s)) destination"))
    eltype(x) == T || throw(ArgumentError(
        "destination stores $T, got $(eltype(x)); convert explicitly"))
    padded = padded_size(s)
    if size(s) == padded
        swizzle_permute!(s, x)
    else
        xp = similar(parent(s), T, padded)
        fill!(xp, zero(T))   # host-side conversion; Microfloat convert can't GPU-compile
        copyto!(view(xp, axes(x)...), x)
        swizzle_permute!(s, xp)
        release!(xp)
    end
    return s
end

@generated function swizzle_permute!(s::SwizzledArray{T,N,Pattern{L,R,C}},
                                     xp::AbstractArray) where {T,N,L,R,C}
    nf = sum(g -> g isa Int ? 1 : length(g), L)
    rflat = Tuple(Iterators.flatten(map(g -> g isa Int ? (g,) : g, R)))
    perm = (rflat..., ntuple(i -> nf + i, N - length(L))...)
    return quote
        dims = (s.sizes..., size(xp)[$(length(L))+1:end]...)
        permutedims!(reshape(parent(s), ($(map(i -> :(dims[$i]), perm)...),)),
                     reshape(xp, dims), $perm)
        return s
    end
end

const SWIZZLES = (;
    f8_4x128 = (; pattern = ((:k1, :k0), (:m1, :m2, :m0), ..) -->
                            (:k1, :m2, :m1, :k0, :m0, ..),
                  context = (; k1 = 4, m2 = 4, m1 = 32)),
)

preset(name::Symbol) = haskey(SWIZZLES, name) ? SWIZZLES[name] :
    throw(ArgumentError("unknown swizzle preset :$name; available: \
                         $(join(map(k -> ":$k", keys(SWIZZLES)), ", "))"))

swizzle(x::AbstractArray, name::Symbol; kws...) =
    (p = preset(name); swizzle(x, p.pattern; kws..., p.context...))
SwizzledArray(x::AbstractArray, name::Symbol; kws...) =
    (p = preset(name); SwizzledArray(x, p.pattern; kws..., p.context...))

# sides may be passed separately, sparing callers Einops' `-->`
swizzle(x::AbstractArray, left, right; kws...) = swizzle(x, left --> right; kws...)
SwizzledArray(x::AbstractArray, left, right; kws...) =
    SwizzledArray(x, left --> right; kws...)

"""
    F8_4x128Array{T,N,X}
    F8_4x128Array(storage; kws...)

Dispatch alias for a [`SwizzledArray`](@ref) carrying the `:f8_4x128`
layout — any naming of it is this exact concrete type. The constructor
form is `SwizzledArray(storage, :f8_4x128; kws...)`.
"""
const F8_4x128Array{T,N,X} = SwizzledArray{T,N,
    Pattern{((1, 2), (3, 4, 5)),           # (k1 k0) (m1 m2 m0)
            (1, 4, 3, 2, 5),               # k1 m2 m1 k0 m0
            (4, nothing, 32, 4, nothing)}, # k1 = 4, m1 = 32, m2 = 4
    X, 5}
F8_4x128Array(x::AbstractArray; kws...) = SwizzledArray(x, :f8_4x128; kws...)

"""
    f8_4x128(x::AbstractArray; kws...)

Shorthand for [`swizzle`](@ref)`(x, :f8_4x128; kws...)`.
"""
f8_4x128(x::AbstractArray; kws...) = swizzle(x, :f8_4x128; kws...)

# ## Element access: decompose each dense index over its left-side group
# (first factor fastest), recompose per the storage groups.

Base.IndexStyle(::Type{<:SwizzledArray}) = IndexCartesian()
@generated function Base.getindex(s::SwizzledArray{T,N,Pattern{L,R,C}},
                                  I::Vararg{Int,N}) where {T,N,L,R,C}
    vals = Dict{Int,Any}()
    stmts = Expr[]
    for (axis, group) in enumerate(L)
        prev = :(I[$axis])
        group isa Int && (vals[group] = prev; continue)
        for f in group[1:end-1]
            q, r = gensym(:q), gensym(:r)
            push!(stmts, :(($q, $r) = fldmod1($prev, s.sizes[$f])))
            vals[f] = r
            prev = q
        end
        vals[group[end]] = prev
    end
    compose(group) = group isa Int ? vals[group] :
        foldr((f, ex) -> :(($ex - 1) * s.sizes[$f] + $(vals[f])), group[1:end-1];
              init=vals[group[end]])
    ix = Any[map(compose, R)..., (:(I[$k]) for k in length(L)+1:N)...]
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
    keys = ntuple(name, sum(g -> g isa Int ? 1 : length(g), L))
    return quote
        y = rearrange(parent(s), $p; NamedTuple{$keys}(s.sizes)...)
        return size(y) == size(s) ? y : y[map(Base.OneTo, size(s))...]  # unpad
    end
end

function Adapt.adapt_structure(to, s::SwizzledArray{T,N,P,X,F}) where {T,N,P,X,F}
    x = adapt(to, parent(s))
    return SwizzledArray{eltype(x),N,P,typeof(x),F}(x, s.dims, s.sizes, s.names)
end

# ## Display: elide the Pattern params; append the pattern and declared sizes

sidestring(side, names) =
    join(map(g -> g isa Int ? String(names[g]) :
                  "(" * join((names[f] for f in g), " ") * ")", side), " ")

function Base.showarg(io::IO, s::SwizzledArray{T,N,Pattern{L,R,C},X,F},
                      toplevel) where {T,N,L,R,C,X,F}
    if isdefined(Base, :make_typealias) && Base.make_typealias(typeof(s)) !== nothing
        show(io, typeof(s))  # an alias (e.g. F8_4x128Array) prints compactly
    else
        print(io, SwizzledArray, "{", T, ", ", N, ", Pattern{…}, ", X, ", ", F, "}")
    end
    toplevel || return
    l, r = sidestring(L, s.names), sidestring(R, s.names)
    N > length(L) && (l *= " ..."; r *= " ...")
    print(io, " with swizzle ")
    printstyled(io, "\"", l, " -> ", r, "\""; color=:green)
    for (j, i) in enumerate([i for i in 1:F if C[i] !== nothing])
        print(io, j == 1 ? " where " : ", ", s.names[i], "=")
        printstyled(io, C[i]; color=:cyan)
    end
    return
end

### Broadcasting
Base.broadcastable(s::SwizzledArray) = copy(s)

Base.print_array(io::IO, s::SwizzledArray) = Base.print_array(io, Base.broadcastable(s))
