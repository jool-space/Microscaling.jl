using Microscaling
using Test

using CUDA
using Random

using BitPacking

function blockscaled_gemm_reference(x_data, x_scale, y_data, y_scale, block_size;
                                    x_block_size=block_size, y_block_size=block_size)
    _bs(b) = b isa Tuple ? b : (b, 1)
    dqX = Float32.(x_data) .* Float32.(repeat(x_scale, inner = _bs(x_block_size)))
    dqY = Float32.(y_data) .* Float32.(repeat(y_scale, inner = _bs(y_block_size)))
    return transpose(dqX) * dqY
end

@testset "Microscaling.jl" begin
    if CUDA.functional()
        include("gemm_cublaslt.jl")
    end
end
