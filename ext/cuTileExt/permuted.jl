using Microscaling

import cuTile as ct
import Adapt

struct PermutedTileArray{T,N,perm,iperm,P<:ct.AbstractTileArray{T,N}} <: ct.AbstractTileArray{T,N}
    parent::P
end

function PermutedTileArray(parent::ct.AbstractTileArray{T,N}, perm::NTuple{N,Int}) where {T,N}
    (isperm(perm) && length(perm) == N) ||
        throw(ArgumentError("invalid permutation $perm for a $N-dimensional array"))
    PermutedTileArray{T,N,perm,invperm(perm),typeof(parent)}(parent)
end

Base.parent(arr::PermutedTileArray) = getfield(arr, :parent)

@generated _genperm(t::Tuple, ::Val{P}) where {P} =
    Expr(:tuple, (:(t[$p]) for p in P)...)

# Deliberately @generated (like cuTile's `Base.transpose(::Tile)`): expansion
# is deferred until the tile type is concrete, so the `permute` intrinsic is
# never reached while inference is still imprecise (pre const-prop), where its
# tfunc would fail on an unshaped `Tile`. The `isconcretetype` check exists to
# make the generator *use* `tile` — deferral only happens for arguments the
# generator consumes.
@generated function _permute(tile, ::Val{P}) where {P}
    isconcretetype(tile) || error("_permute expanded with non-concrete tile type $tile")
    return :(permutedims(tile, $P))
end

Base.size(arr::PermutedTileArray{T,N,perm}) where {T,N,perm} =
    _genperm(size(parent(arr)), Val(perm))
Base.size(arr::PermutedTileArray{T,N,perm}, d::Integer) where {T,N,perm} =
    size(parent(arr), perm[d])

function Adapt.adapt_structure(to::ct.KernelAdaptor,
        arr::PermutedDimsArray{T,N,perm,<:Any,<:BlockscaledArray{T,N}}) where {T,N,perm}
    return PermutedTileArray(Adapt.adapt(to, parent(arr)), perm)
end

# `order=o` follows cuTile's load/store convention: `index[i]`/`shape[i]`
# describe tile dim `i`, which maps to dim `o[i]` of *this* (permuted) array,
# i.e. parent dim `perm[o[i]]`. Compose both into one parent-dim mapping `q`
# (`o === nothing` leaves `q = perm`) instead of forwarding `order` to the
# parent load, whose composite implementations (e.g. `BlockscaledTileArray`)
# index per-dimension state like `block_size` positionally and are not
# order-aware.
@generated function _compose(::Val{perm}, ::Val{order}) where {perm, order}
    order === nothing && return :(($(Val(perm)), $(Val(invperm(perm)))))
    (order isa NTuple{length(perm), Int} && isperm(order)) ||
        return :(throw(ArgumentError($(string("invalid order ", order,
                                              " for permutation ", perm)))))
    q = map(o -> perm[o], order)
    return :(($(Val(q)), $(Val(invperm(q)))))
end

function ct.load(arr::PermutedTileArray{T,N,perm}, index, shape;
                 order::Union{NTuple{N,Int}, Nothing}=nothing,
                 kws...) where {T,N,perm}
    q, qi = _compose(Val(perm), Val(order))
    tile = ct.load(parent(arr), _genperm(index, qi), _genperm(shape, qi); kws...)
    return _permute(tile, q)
end

function ct.store(arr::PermutedTileArray{T,N,perm}, index, tile;
                  order::Union{NTuple{N,Int}, Nothing}=nothing,
                  kws...) where {T,N,perm}
    _, qi = _compose(Val(perm), Val(order))
    return ct.store(parent(arr), _genperm(index, qi), _permute(tile, qi); kws...)
end
