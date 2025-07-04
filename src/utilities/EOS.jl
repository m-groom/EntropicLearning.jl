# Utility functions for Entropic Outlier Sparsification (EOS)
# This file contains standalone functions for computing EOS weights
# without the full iterative fitting procedure

export eos_weights, eos_outlier_scores, calculate_eos_weights, eos_distances, supports_eos

# ==============================================================================
# Distance Function Protocol
# ==============================================================================

"""
    eos_distances(model, fitresult, X, [y])

Compute distances/losses for each sample in X using the fitted model.

This function must be implemented for any model to be EOS-compatible.
For supervised models, y may be provided for computing supervised losses.

# Arguments
- `model`: The MLJ model instance
- `fitresult`: The result from fitting the model
- `X`: Input data
- `y`: Target data (optional, for supervised models)

# Returns
- Vector of distances/losses, one per sample

# Example Implementation
```julia
# For a classifier using cross-entropy loss
function EntropicLearning.eos_distances(model::MyClassifier, fitresult, X, y=nothing)
    if isnothing(y)
        # For transform: use entropy of predicted probabilities
        probs = predict(model, fitresult, X)
        return -[sum(p .* log.(p .+ eps()) for p in probs]
    else
        # For fit: use cross-entropy loss
        probs = predict(model, fitresult, X)
        # ... compute cross-entropy with y
    end
end
```
"""
function eos_distances end

# Default error message
eos_distances(model, args...) = 
    error("Model type $(typeof(model)) must implement eos_distances to be EOS-compatible. " *
          "See ?eos_distances for details.")

"""
    supports_eos(::Type{ModelType})

Check if a model type supports EOS by implementing the required interface.

This function automatically returns `true` if `eos_distances` is implemented
for the given model type.
"""
function supports_eos(::Type{M}) where M
    # Check if there's a method for eos_distances with this model type
    # We check for methods with 3 or 4 arguments (X, or X and y)
    method_exists = false
    
    # Check for eos_distances(model::M, fitresult, X)
    sig3 = Tuple{typeof(eos_distances), M, Any, Any}
    method_exists |= !isempty(methods(eos_distances, sig3))
    
    # Check for eos_distances(model::M, fitresult, X, y)  
    sig4 = Tuple{typeof(eos_distances), M, Any, Any, Any}
    method_exists |= !isempty(methods(eos_distances, sig4))
    
    return method_exists
end

# ==============================================================================
# Core EOS Functions
# ==============================================================================

"""
    eos_weights(distances, α)

Calculate EOS weights from distances using the closed-form solution from
Theorem in Horenko (2022).

# Arguments
- `distances`: Vector of distances/losses for each sample
- `α`: Entropic regularization parameter (>0)

# Returns
- Vector of weights in [0,1] that sum to 1

# Details
The weights are computed as:
```
wᵢ = exp(-dᵢ/α) / Σⱼ exp(-dⱼ/α)
```

Uses log-sum-exp trick for numerical stability.
"""
function eos_weights(distances::AbstractVector{<:Real}, α::Real)
    α > 0 || error("α must be positive")
    
    # Handle edge cases
    n = length(distances)
    n > 0 || return Float64[]
    
    # All distances equal -> uniform weights
    if all(d -> d ≈ first(distances), distances)
        return fill(1/n, n)
    end
    
    # Log-sum-exp trick for numerical stability
    log_weights = -distances / α
    max_log_weight = maximum(log_weights)
    exp_weights = exp.(log_weights .- max_log_weight)
    weights = exp_weights / sum(exp_weights)
    
    return weights
end

"""
    calculate_eos_weights(model, fitresult, X, α; y=nothing)

Calculate EOS weights for data X using a fitted model.

This is the main utility function for mode 2 (single-step weight calculation).

# Arguments
- `model`: A fitted MLJ model that implements `eos_distances`
- `fitresult`: The result from fitting the model
- `X`: Input data to calculate weights for
- `α`: Entropic regularization parameter
- `y`: Target data (optional, for supervised losses)

# Returns
- Vector of weights in [0,1]

# Example
```julia
# Fit any MLJ model
mach = machine(SomeModel(), X, y) |> fit!

# Calculate EOS weights for the training data
weights = calculate_eos_weights(
    mach.model, 
    mach.fitresult, 
    X, 
    1.0;  # α parameter
    y=y
)
```
"""
function calculate_eos_weights(model, fitresult, X, α::Real; y=nothing)
    if !supports_eos(typeof(model))
        error("Model type $(typeof(model)) must implement eos_distances. " *
              "See ?eos_distances for implementation details.")
    end
    
    distances = eos_distances(model, fitresult, X, y)
    return eos_weights(distances, α)
end

"""
    eos_outlier_scores(model, fitresult, X, α; y=nothing)

Calculate outlier scores (1 - weight) for data X using a fitted model.

Higher scores indicate more outlying samples.

# Arguments
- `model`: A fitted MLJ model that implements `eos_distances`
- `fitresult`: The result from fitting the model  
- `X`: Input data to score
- `α`: Entropic regularization parameter
- `y`: Target data (optional)

# Returns
- Vector of outlier scores in [0,1]
"""
function eos_outlier_scores(model, fitresult, X, α::Real; y=nothing)
    weights = calculate_eos_weights(model, fitresult, X, α; y=y)
    return 1 .- weights
end
