module EntropicLearning

include("utilities/Transformers.jl")
using .Transformers
export MinMaxScaler, QuantileTransformer

using SparseArrays
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

include("eSPA/eSPA.jl")
using .eSPAClassifier
export eSPA

end # module EntropicLearning
