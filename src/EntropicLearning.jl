module EntropicLearning

include("utilities/Transformers.jl")
using .Transformers
export MinMaxScaler, QuantileTransformer

using SparseArrays
using Random
using Tables
include("common/functions.jl")

# Include eSPA model
include("eSPA/eSPA.jl")
using .eSPA
export eSPAClassifier

# Common package metadata
const PKG_METADATA = (
    package_name="EntropicLearning",
    package_uuid="857d3a31-ba67-457f-9b14-0a8f313fa218",
    package_url="https://github.com/m-groom/EntropicLearning.jl",
    package_license="ASL",
)

using MLJModelInterface
MLJModelInterface.metadata_pkg.(
    (eSPAClassifier, MinMaxScaler, QuantileTransformer);
    PKG_METADATA...,
    is_pure_julia=true,
    is_wrapper=false,
)

end # module EntropicLearning
