using Microscaling
using Microscaling: elements, scales, block_size, scale_type, element_type
using Einops
using Test

using CUDACore
using cuBLASLt
import cuTile as ct
using Random

using BitPacking
import Adapt

@assert !isnothing(Base.get_extension(BitPacking, :cuTileExt))

dequantize(scales, elements, block) =
    Float32.(elements) .* repeat(Float32.(scales); inner = block)

function blockscaled_gemm_reference(x_data, x_scale, y_data, y_scale, block_size;
                                    x_block_size=block_size, y_block_size=block_size)
    _bs(b) = b isa Tuple ? b : (b, 1)
    dqX = Float32.(x_data) .* Float32.(repeat(x_scale, inner = _bs(x_block_size)))
    dqY = Float32.(y_data) .* Float32.(repeat(y_scale, inner = _bs(y_block_size)))
    return transpose(dqX) * dqY
end

@testset "Microscaling.jl" begin
    include("blockscaling.jl")
    include("swizzle.jl")

    if CUDACore.functional()
        # the cuTile gemms lean on Blackwell (narrow-type loads, tcgen05
        # block-scaled MMA); gemm_cublaslt.jl carries its own capability gates
        if CUDACore.capability(CUDACore.device()) >= v"10.0"
            include("gemm_agnostic.jl")
            include("gemm_mxfp8.jl")
        else
            @info "skipping cuTile gemm testsets (CC ≥ 10.0 required)"
        end
        include("gemm_cublaslt.jl")
        include("gemm_cudnn.jl")
    end
end
