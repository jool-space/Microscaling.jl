module cuBLASLtExt

using Microscaling:
    F8_4x128Array,
    Float8_E4M3FN, Float8_E8M0FNU,
    BlockscaledArray,
    block_size, scale_type, elements, scales

import cuBLASLt

cuBLASLt.ltptr(A::F8_4x128Array) = cuBLASLt.ltptr(parent(A))

cuBLASLt.ltdata(A::BlockscaledArray) = elements(A)
cuBLASLt.ltscale(A::BlockscaledArray) = scales(A)

function cuBLASLt.scale_mode(A::BlockscaledArray{<:Any,N}) where {N}
    2 <= N <= 3 || throw(ArgumentError(
        "cuBLASLt matmul takes matrices or 3-dimensional batches; got $(N)D"))
    mode = _scale_mode(block_size(A), scale_type(A))
    check_scale_layout(scales(A), mode)
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

# cuBLASLt only ever sees a raw scale pointer, so it cannot verify layouts
# itself: mislaid scales are silently misread, or read out of bounds. Each
# mode's memory contract is enforced here instead.
#
# The MX modes read through a swizzled 128×4-entry tiled layout; an F8_4x128Array
# is swizzled and tile-padded by construction. The Hopper f32 modes read plain
# arrays, but with fixed orientations (cuBLAS docs, "Scaling factors
# layouts"): :vec128_f32 scales are OUTER-major — memory shape outer × K÷128,
# the transpose of the (K÷128, outer) block-scale geometry — and
# :blk128x128_f32 scales are K-major with the stride between outer-block
# columns a multiple of 4 (pad K÷128 up to 4). Both coincide with a plain
# column-major scale array only in the degenerate single-column cases, which
# is exactly why a mislaid array cannot be caught by shape alone.
function check_scale_layout(x, mode::Symbol)
    if mode in (:vec32_ue8m0, :vec16_ue4m3)
        x isa F8_4x128Array || throw(ArgumentError(
            "cuBLASLt accesses :$mode scales through a swizzled tiled layout, " *
            "but the scales are a $(nameof(typeof(x))); wrap them with `f8_4x128`"))
    elseif mode === :vec128_f32
        outer_major = (size(x, 2) == 1 || stride(x, 2) == 1) &&
                      (size(x, 1) == 1 || stride(x, 1) == size(x, 2))
        outer_major || throw(ArgumentError(
            "cuBLASLt reads :vec128_f32 scales outer-major (dense outer × K÷128 " *
            "in memory), but this scale array is K-major; store the scales " *
            "transposed, e.g. `transpose(outer_by_kblocks)`"))
    elseif mode === :blk128x128_f32
        padded = stride(x, 1) == 1 && (size(x, 2) == 1 || stride(x, 2) % 4 == 0)
        padded || throw(ArgumentError(
            "cuBLASLt reads :blk128x128_f32 scales K-major with the stride " *
            "between outer-block columns a multiple of 4; pad the K÷128 " *
            "dimension up to a multiple of 4 and pass a view of the padding"))
    end
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
