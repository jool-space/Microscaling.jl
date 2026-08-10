module cuBLASLtExt

using Microscaling:
    Sm1xxArray,
    Float8_E4M3FN, Float8_E8M0FNU,
    BlockscaledArray,
    block_size, scale_type

import cuBLASLt

cuBLASLt.ltptr(A::Sm1xxArray) = cuBLASLt.ltptr(parent(A))

cuBLASLt.ltdata(A::BlockscaledArray) = A.p
cuBLASLt.ltscale(A::BlockscaledArray) = A.x

function cuBLASLt.scale_mode(A::BlockscaledArray{<:Any,N}) where {N}
    2 <= N <= 3 || throw(ArgumentError(
        "cuBLASLt matmul takes matrices or 3-dimensional batches; got $(N)D"))
    mode = _scale_mode(block_size(A), scale_type(A))
    check_swizzled(A.x, mode)
    return mode
end

blockstr(t::Tuple) = "(" * join((b isa Colon ? ":" : b for b in t), ",") * ")"

# Both gemm operands store K in dimension 1 and the outer M/N in dimension 2,
# so the (K, outer) block geometry plus the scale type picks the mode;
# dimension 3, when present, is the batch. The per-matrix modes repeat their
# scale layout each batch entry (batch block 1); a (:,:) scalar is tensorwide
# when its block spans the batch too, per-entry when it blocks by 1.
function _scale_mode(block::Tuple, ::Type{S}) where {S}
    k, outer = block[1], block[2]
    batch = length(block) == 3 ? block[3] : nothing
    if k isa Colon && outer isa Colon
        S === Float32 || throw(ArgumentError(
            "scalar scales expect Float32, got $S"))
        (batch === nothing || batch isa Colon) && return :scalar_f32
        batch == 1 && return :per_batch_scalar_f32
        throw(ArgumentError(
            "scalar scales must span the whole batch (:) or block it by 1; " *
            "got $(blockstr(block))"))
    end
    batch === nothing || batch == 1 || throw(ArgumentError(
        "batched block scales must repeat per batch entry (batch block 1); " *
        "got $(blockstr(block))"))
    return _matrix_scale_mode(k, outer, S)
end

# cuBLASLt accesses block-mode scales through its swizzled 128×4-entry tiled
# layout, but only ever sees a raw pointer, so it cannot verify the layout
# itself: unswizzled scales are silently misread, or read out of bounds. An
# Sm1xxArray is swizzled and tile-padded by construction.
function check_swizzled(x, mode::Symbol)
    mode in (:vec32_ue8m0, :vec16_ue4m3) || return nothing
    x isa Sm1xxArray || throw(ArgumentError(
        "cuBLASLt accesses :$mode scales through a swizzled tiled layout, " *
        "but the scales are a $(nameof(typeof(x))); wrap them with `sm1xx`"))
    return nothing
end

function _matrix_scale_mode(k, m, ::Type{Float8_E8M0FNU})
    k == 32 && m == 1 || throw(ArgumentError(
        "UE8M0 scales expect (32,1) blocks, got $(blockstr((k, m)))"))
    return :vec32_ue8m0
end

function _matrix_scale_mode(k, m, ::Type{Float8_E4M3FN})
    k == 16 && m == 1 || throw(ArgumentError(
        "UE4M3 scales expect (16,1) blocks, got $(blockstr((k, m)))"))
    return :vec16_ue4m3
end

function _matrix_scale_mode(k, m, ::Type{Float32})
    if k isa Colon
        m == 1 || throw(ArgumentError(
            "K-wide Float32 scales expect (:,1) blocks (one scale per outer " *
            "element), got $(blockstr((k, m)))"))
        return :outer_vec_f32
    end
    k == 128 || throw(ArgumentError("Float32 scales expect k=128 or k=:, got $k"))
    m == 1 && return :vec128_f32
    m == 128 && return :blk128x128_f32
    throw(ArgumentError("Float32 scales expect m=1 or m=128, got $m"))
end

_matrix_scale_mode(k, m, ::Type{S}) where {S} = throw(ArgumentError(
    "no cuBLASLt scale mode for $S scales with $(blockstr((k, m))) blocks"))

end
