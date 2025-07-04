module EntropicLearning

include("utilities/Transformers.jl")
using .Transformers
export MinMaxScaler, QuantileTransformer

using SparseArrays
include("common/functions.jl")

# Include EOS utility functions
using MLJModelInterface
include("utilities/eos.jl")
export eos_weights, eos_distances, calculate_eos_weights, eos_outlier_scores

include("eSPA/eSPA.jl")
using .eSPA
export eSPAClassifier

# # Include EOS wrapper model
# include("EOS/EOS.jl")
# using .EOS
# export EOSWrapper

# Common package metadata
const PKG_METADATA = (
    package_name="EntropicLearning",
    package_uuid="857d3a31-ba67-457f-9b14-0a8f313fa218",
    package_url="https://github.com/m-groom/EntropicLearning.jl",
    is_pure_julia=true,
    package_license="ASL",
)

MLJModelInterface.metadata_pkg.(
    (eSPAClassifier, MinMaxScaler, QuantileTransformer);
    PKG_METADATA...,
    is_wrapper=false,
)

# MLJModelInterface.metadata_pkg(
#     EOSWrapper;
#     PKG_METADATA...,
#     is_wrapper=true,  # EOS is a wrapper model
# )

end # module EntropicLearning
