# Example usage of EOS in EntropicLearning.jl

# This example shows how to use EOS in two modes:
# Mode 1: As a wrapper model (full iterative EOS)
# Mode 2: As utility functions for calculating weights on pre-fitted models

using EntropicLearning
using MLJBase
using Random

# Generate some example data
Random.seed!(123)
n = 100
X = randn(n, 5)
y = rand([0, 1], n)

# ==============================================================================
# Mode 1: EOS as a Wrapper Model
# ==============================================================================

# First, you need a model that:
# 1. Supports sample weights
# 2. Implements eos_distances

# Example implementation for a mock classifier:
struct MockClassifier <: MLJModelInterface.Deterministic end

MLJModelInterface.fit(::MockClassifier, verbosity, X, y; weights=nothing) = 
    (fitresult=(X=X, y=y, weights=weights), cache=nothing, report=nothing)

MLJModelInterface.predict(::MockClassifier, fitresult, Xnew) = 
    fill(0.5, MLJModelInterface.nrows(Xnew))  # Mock predictions

MLJModelInterface.supports_weights(::Type{<:MockClassifier}) = true

# Implement eos_distances for the mock classifier
function EntropicLearning.eos_distances(model::MockClassifier, fitresult, X, y=nothing)
    # Return some mock distances (e.g., distance from mean)
    n = size(X, 1)
    return rand(n) .+ 0.1  # Mock distances
end

# Now we can use EOS wrapper
mock_model = MockClassifier()
eos_model = EOS(mock_model, α=1.0)

# Train the EOS-wrapped model
# mach = machine(eos_model, X, y)
# fit!(mach)

# Get predictions and outlier scores
# ŷ = predict(mach, X)
# outlier_scores = transform(mach, X)

# ==============================================================================
# Mode 2: EOS Utility Functions
# ==============================================================================

# For any pre-fitted model that implements eos_distances,
# you can calculate EOS weights directly:

# First, fit the model normally
# mach = machine(mock_model, X, y)
# fit!(mach)

# Then calculate EOS weights
# weights = calculate_eos_weights(
#     mach.model,
#     mach.fitresult,
#     X,
#     1.0;  # α parameter
#     y=y   # optional for supervised losses
# )

# Or get outlier scores directly
# outlier_scores = eos_outlier_scores(
#     mach.model,
#     mach.fitresult,
#     X,
#     1.0
# )

# ==============================================================================
# Implementing eos_distances for Your Model
# ==============================================================================

# To make any MLJ model EOS-compatible, implement:
#
# function EntropicLearning.eos_distances(
#     model::YourModelType, 
#     fitresult, 
#     X, 
#     y=nothing
# )
#     # Return a vector of distances/losses (one per sample)
#     # For regression: squared residuals
#     # For classification: negative log-likelihood or cross-entropy
#     # For clustering: distance to nearest center
#     # etc.
# end

# Once implemented, supports_eos(YourModelType) will automatically return true!
