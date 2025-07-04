module EntropicLearning

include("utilities/Transformers.jl")
using .Transformers
export MinMaxScaler, QuantileTransformer

using SparseArrays
include("common/functions.jl")

include("eSPA/eSPA.jl")
using .eSPA
export eSPAClassifier

include("utilities/EOS.jl")
using .EOS
export EOS, eos_weights, eos_outlier_scores, calculate_eos_weights, eos_distances, supports_eos

using MLJModelInterface
MLJModelInterface.metadata_pkg.(
    (eSPAClassifier, MinMaxScaler, QuantileTransformer),
    package_name="EntropicLearning",
    package_uuid="857d3a31-ba67-457f-9b14-0a8f313fa218",
    package_url="https://github.com/m-groom/EntropicLearning.jl",
    is_pure_julia=true,
    package_license="ASL",
    is_wrapper=false,
)

MLJModelInterface.metadata_pkg(
    EOS,
    package_name="EntropicLearning",
    package_uuid="857d3a31-ba67-457f-9b14-0a8f313fa218",
    package_url="https://github.com/m-groom/EntropicLearning.jl",
    is_pure_julia=true,
    package_license="ASL",
    is_wrapper=true,  # EOS is a wrapper model
)

end # module EntropicLearning
