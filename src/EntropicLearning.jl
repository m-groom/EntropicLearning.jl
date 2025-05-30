module EntropicLearning

using SparseArrays

include("utilities/Transformers.jl")
using .Transformers
export MinMaxScaler, QuantileTransformer

include("common/functions.jl")
export safelog, entropy, assign_closest, assign_closest!, left_stochastic, left_stochastic!, right_stochastic, right_stochastic!, smallest, smaller, small

end # module EntropicLearning
