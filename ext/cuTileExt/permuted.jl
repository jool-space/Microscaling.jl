using Microscaling

import cuTile as ct
import Adapt

"""
    PermutedTileArray{T,N,perm,iperm,P<:ct.AbstractTileArray{T,N}}

Lazy dimension permutation of an arbitrary `AbstractTileArray`, mirroring
`Base.PermutedDimsArray`. `ct.load`/`ct.store` translate the logical (permuted)
index and shape into the parent's physical dimension order, and permute the
tile itself, so kernels can stay layout-agnostic.

Plain `TileArray`s don't need this (their strides already encode any
permutation); it exists for wrapper tile arrays whose layout is not expressible
with strides, e.g. sub-byte packed elements or swizzled scale factors.
"""
struct PermutedTileArray{T,N,perm,iperm,P<:ct.AbstractTileArray{T,N}} <: ct.AbstractTileArray{T,N}
    parent::P
end

function PermutedTileArray{perm,iperm}(parent::ct.AbstractTileArray{T,N}) where {T,N,perm,iperm}
    (isperm(perm) && length(perm) == N && Tuple(invperm(perm)) == iperm) ||
        throw(ArgumentError("invalid permutation $perm / $iperm for a $N-dimensional array"))
    PermutedTileArray{T,N,perm,iperm,typeof(parent)}(parent)
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
        arr::PermutedDimsArray{T,N,perm,iperm,<:BlockscaledArray{T,N}}) where {T,N,perm,iperm}
    return PermutedTileArray{perm,iperm}(Adapt.adapt(to, parent(arr)))
end

function ct.load(arr::PermutedTileArray{T,N,perm,iperm}, index, shape;
                 kws...) where {T,N,perm,iperm}
    tile = ct.load(parent(arr), _genperm(index, Val(iperm)), _genperm(shape, Val(iperm)); kws...)
    return _permute(tile, Val(perm))
end

function ct.store(arr::PermutedTileArray{T,N,perm,iperm}, index, tile;
                  kws...) where {T,N,perm,iperm}
    return ct.store(parent(arr), _genperm(index, Val(iperm)), _permute(tile, Val(iperm)); kws...)
end
