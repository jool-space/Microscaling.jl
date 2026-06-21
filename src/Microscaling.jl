module Microscaling

using Republic

include("blockscaling.jl")

@reexport using Narrow
@reexport using Microfloats
@reexport using .Blockscaling

include("sm1xx.jl")
export Sm1xxArray, sm1xx

end
