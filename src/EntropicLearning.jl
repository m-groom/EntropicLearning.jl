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
    normalise!,
    softmax,
    softmax!,
    smallest,
    smaller,
    small

include("eSPA/eSPA.jl")
using .eSPA
export eSPAClassifier

using MLJModelInterface
MLJModelInterface.metadata_pkg(
    eSPAClassifier,
    package_name="EntropicLearning",
    package_uuid="857d3a31-ba67-457f-9b14-0a8f313fa218",
    package_url="https://github.com/m-groom/EntropicLearning.jl",
    is_pure_julia=true,
    package_license="ASL",
    is_wrapper=false,
)

end # module EntropicLearning
