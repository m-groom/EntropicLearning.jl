# Utility functions for Entropic Outlier Sparsification (EOS)
# This file contains standalone functions for computing EOS weights
# without the full iterative fitting procedure

# ==============================================================================
# Distance Function Protocol
# ==============================================================================

"""
    eos_distances(model, fitresult, X, [y])

Compute distances/losses for each sample in X using the fitted model.

This function must be implemented for any model to be EOS-compatible.
For unsupervised models, the function signature is:
    `eos_distances(model, fitresult, X)`
For supervised models, the function signature is:
    `eos_distances(model, fitresult, X, y)`

# Arguments
- `model`: The MLJ model instance
- `fitresult`: The result from fitting the model
- `X`: Input data
- `y`: Target data (optional, for supervised models)

# Returns
- Vector of distances/losses, one per sample

# Example Implementation
```julia
function EntropicLearning.eos_distances(model::eSPAClassifier, fitresult, X, y=nothing)
    # Extract the model parameters from the fitresult
    C = fitresult.C
    W = fitresult.W
    G = fitresult.G

    # Return the discretisation error (per sample)
    return sum(W .* (X' .- C * G) .^ 2, dims=1)
end
```
"""
function eos_distances end

# Default error message
function eos_distances(model, args...)
    return error(
        "Model type $(typeof(model)) must implement eos_distances to be EOS-compatible. " *
        "See ?eos_distances for details.",
    )
end

# ==============================================================================
# Core EOS Functions
# ==============================================================================

"""
    eos_weights(distances, alpha)

Calculate EOS weights from distances using the closed-form solution from Horenko (2022).

# Arguments
- `distances`: Vector of distances/losses for each sample
- `alpha`: Entropic regularisation parameter (>0)

# Returns
- Vector of weights in [0,1] that sum to 1
"""
function eos_weights(distances::AbstractVector{Tr}, alpha::Real) where {Tr<:Real}
    # Check for valid alpha
    alpha > 0 || error("alpha must be positive")

    # Floating point type for weights
    Tf = float(Tr)

    # Handle edge cases
    T_instances = length(distances)
    T_instances > 0 || return Tf[]

    # Initialise weights
    weights = zeros(Tf, T_instances)

    # Call mutating function to update weights
    update_weights!(weights, distances, alpha)

    return weights
end

# Update step for the sample weights vector
function update_weights!(
    weights::AbstractVector{Tf}, distances::AbstractVector{Tr}, alpha::Real
) where {Tf<:AbstractFloat,Tr<:Real}
    # Get dimensions
    T_instances = length(distances)

    if isfinite(alpha)
        softmax!(weights, -distances; prefactor=Tf(alpha))
    else
        # Set weights to the uniform distribution
        fill!(weights, Tf(1.0) / T_instances)
    end
    return nothing
end

"""
    calculate_eos_weights(model, fitresult, X, alpha; y=nothing)

Calculate EOS weights for data X using a fitted model.

# Arguments
- `model`: A fitted MLJ model that implements `eos_distances`
- `fitresult`: The result from fitting the model
- `X`: Input data to calculate weights for
- `alpha`: Entropic regularisation parameter
- `y`: Target data (optional, for supervised losses)

# Returns
- Vector of weights in [0,1] that sum to 1

"""
function calculate_eos_weights(model, fitresult, X, alpha::Real; y=nothing)
    distances = if MMI.is_supervised(model)
        eos_distances(model, fitresult, X, y)
    else
        eos_distances(model, fitresult, X)
    end
    return eos_weights(distances, alpha)
end

"""
    eos_outlier_scores(model, fitresult, X, alpha; y=nothing)

Calculate outlier scores (1 - weight) for data X using a fitted model.

Higher scores indicate more outlying samples.

# Arguments
- `model`: A fitted MLJ model that implements `eos_distances`
- `fitresult`: The result from fitting the model
- `X`: Input data to score
- `alpha`: Entropic regularisation parameter
- `y`: Target data (optional)

# Returns
- Vector of outlier scores in [0,1] that sum to 1
"""
function eos_outlier_scores(model, fitresult, X, alpha::Real; y=nothing)
    weights = calculate_eos_weights(model, fitresult, X, alpha; y=y)
    return 1 .- weights
end
