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

"""
    eos_weights(distances, alpha_range, target_Deff; <kwargs>)

Calculate EOS weights by searching for an `alpha` that yields  a target effective
dimension `target_Deff`.

This method finds the `alpha` within `alpha_range` that produces `eos_weights` with
the desired `target_Deff` (effective dimension).

# Arguments
- `distances::AbstractVector{<:Real}`: Vector of sample distances/losses.
- `alpha_range::Tuple{<:Real,<:Real}`: The search range for `alpha`.
- `target_Deff::Real=0.5`: The target effective dimension.

# Keyword Arguments
- `normalise::Bool=true`: Whether `target_Deff` is normalised (i.e., a value between
  `1/length(distances)` and 1).
- `atol::Real=1e-6`: Absolute tolerance for convergence checking (|objective(found_alpha)| < atol).
- `kwargs...`: Additional keyword arguments passed to `Roots.find_zero` (e.g., `maxiters`, `xatol`, `xrtol`).

# Returns
- `(weights, alpha)`: A tuple containing the calculated `weights` and the found `alpha`.
"""
function eos_weights(
    distances::AbstractVector{<:Real},
    alpha_range::Tuple{<:Real,<:Real},
    target_Deff::Real=0.5;
    normalise::Bool=true,
    atol::Real=1e-6,
    kwargs...,
)
    # Validate alpha_range
    alpha_range[1] > 0 && alpha_range[2] > 0 || error("alpha_range must contain positive values")
    @assert alpha_range[1] < alpha_range[2] "alpha_range must be a valid range (first value must be less than second)"

    # Pre-allocate a weights vector to be reused inside the objective function.
    Tf = float(eltype(distances))
    weights = zeros(Tf, length(distances))

    # Objective function: find alpha where current_Deff - target_Deff is zero
    function objective(alpha::T) where {T<:Real}
        update_weights!(weights, distances, alpha)
        current_Deff = effective_dimension(weights; normalise=normalise)
        return current_Deff - target_Deff
    end

    # Find the alpha that solves the objective function
    found_alpha = Roots.find_zero(objective, alpha_range, Roots.Chandrapatla(); atol=atol, kwargs...)

    # Check convergence
    final_residual = abs(objective(found_alpha))
    if final_residual > atol
        error("Root finding failed to converge. Final residual: $final_residual > $atol. " *
              "Try increasing maxiters or relaxing atol.")
    end

    # Return the final weights and the alpha that produced them
    return (weights=eos_weights(distances, found_alpha), alpha=found_alpha)
end

"""
    calculate_eos_weights(model, fitresult, X, alpha_range, target_Deff; <kwargs>)

Calculate EOS weights for a model by searching for an `alpha` that yields a target effective
dimension `target_Deff`.

This convenience wrapper first computes `distances` using the provided model, then calls
the corresponding `eos_weights` method to find the `alpha` that matches the `target_Deff`.

# Arguments
- `model`: A fitted MLJ model that implements `eos_distances`.
- `fitresult`: The result from fitting the model.
- `X`: Input data.
- `alpha_range::Tuple{<:Real,<:Real}`: The search range for `alpha`.
- `target_Deff::Real=0.5`: The target effective dimension for the weights.

# Keyword Arguments
- `y=nothing`: Target data (for supervised models).
- `kwargs...`: Additional keyword arguments forwarded to `eos_weights` (e.g., `normalise`, `atol`, `maxiters`).
  See `?eos_weights` for details.

# Returns
- `(weights, alpha)`: A tuple with the calculated `weights` and the found `alpha`.
"""
function calculate_eos_weights(
    model,
    fitresult,
    X,
    alpha_range::Tuple{<:Real,<:Real},
    target_Deff::Real=0.5;
    y=nothing,
    kwargs...,
)
    distances = if MMI.is_supervised(model)
        eos_distances(model, fitresult, X, y)
    else
        eos_distances(model, fitresult, X)
    end

    return eos_weights(
        distances,
        alpha_range,
        target_Deff;
        kwargs...,
    )
end

"""
    eos_outlier_scores(model, fitresult, X, alpha_range, target_Deff; <kwargs>)

Calculate EOS outlier scores by searching for an `alpha` that yields a target effective
dimension `target_Deff`.

This method finds the `alpha` that produces EOS weights with a specific effective
dimension, and then computes the corresponding outlier scores (`1 .- weight`). It serves as a
convenience wrapper around `calculate_eos_weights`.

# Arguments
- `model`: A fitted MLJ model that implements `eos_distances`.
- `fitresult`: The result from fitting the model.
- `X`: Input data.
- `alpha_range::Tuple{<:Real,<:Real}`: The search range for `alpha`.
- `target_Deff::Real=0.5`: The target effective dimension for the *weights*.

# Keyword Arguments
- `y=nothing`: Target data (for supervised models).
- `kwargs...`: Additional keyword arguments forwarded to `eos_weights` (e.g., `normalise`, `atol`, `maxiters`).
  See `?eos_weights` for details.

# Returns
- `(scores, alpha)`: A tuple with the calculated outlier scores and the found `alpha`.
"""
function eos_outlier_scores(
    model,
    fitresult,
    X,
    alpha_range::Tuple{<:Real,<:Real},
    target_Deff::Real=0.5;
    y=nothing,
    kwargs...,
)
    # Find the appropriate weights and alpha by calling the corresponding eos_weights function.
    result = calculate_eos_weights(
        model,
        fitresult,
        X,
        alpha_range,
        target_Deff;
        y=y,
        kwargs...,
    )

    # Calculate outlier scores and return with the found alpha.
    return (scores=(1 .- result.weights), alpha=result.alpha)
end
