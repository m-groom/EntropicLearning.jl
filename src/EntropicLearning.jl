module EntropicLearning

using SparseArrays

include("utilities/Transformers.jl")
using .Transformers
export MinMaxScaler, QuantileTransformer

include("common/functions.jl")
export safelog,
    entropy,
    cross_entropy,
    assign_closest,
    assign_closest!,
    left_stochastic,
    left_stochastic!,
    right_stochastic,
    right_stochastic!,
    softmax,
    softmax!,
    smallest,
    smaller,
    small

end # module EntropicLearning
