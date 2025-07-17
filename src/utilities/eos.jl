# Utility functions for Entropic Outlier Sparsification (EOS)
# This file contains standalone functions for computing EOS weights
# without the full iterative fitting procedure

# ==============================================================================
# Distance Function Protocol
# ==============================================================================

"""
    eos_distances(model, fitresult, X, args...)

Compute distances/losses for each sample in X using the fitted model.

This function must be implemented for any model to be EOS-compatible.
For unsupervised models, the function signature is:
    `eos_distances(model, fitresult, X)`
For supervised models, the function signature is:
    `eos_distances(model, fitresult, X, args...)`

# Arguments
- `model`: The MLJ model instance
- `fitresult`: The result from fitting the model
- `X`: Input data
- `args...`: Additional arguments (optional, for supervised models)

# Returns
- Vector of distances/losses, one per sample

# Example Implementation
```julia
function EntropicLearning.eos_distances(model::eSPAClassifier, fitresult, X, args...)
    # Extract the model parameters from the fitresult
    C = fitresult.C
    W = fitresult.W
    G = fitresult.G

    # Return the discretisation error (per sample)
    return sum(W .* (X .- C * G) .^ 2, dims=1)
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

"""
    eos_loss(model, distances, weights, fitresult, args...)

Compute the total EOS loss for a fitted model.

This function calculates the overall loss for the Entropic Outlier Sparsification (EOS)
algorithm. While the default implementation is the weighted sum of sample distances
(`dot(weights, distances)`), models can override this method to incorporate custom
loss calculations, such as including additional regularisation terms that depend on the
sample weights.

This is particularly useful when the loss function for a model is more complex than
a simple sum of per-sample distances.

# Arguments
- `model`: The MLJ model instance.
- `distances::AbstractVector`: Vector of distances/losses for each sample, calculated from
  `eos_distances`.
- `weights::AbstractVector`: Vector of EOS weights for each sample, calculated from `eos_weights`.
- `fitresult`: The result from fitting the model.
- `args...`: The original data (`X`, `y`, etc.) that was used for fitting. This allows for
   loss calculations that depend on the original data.

# Returns
- A scalar value representing the total loss.

# Default Implementation
```julia
function eos_loss(model, distances::AbstractVector, weights::AbstractVector, fitresult, args...)
    return dot(weights, distances)
end
```
# See Also
- [`eos_distances`](@ref): Compute distances/losses for each sample in X using the fitted model.
- [`eos_weights`](@ref): Calculate EOS weights from distances using the closed-form solution
  from Horenko (2022).

"""
function eos_loss end

# Default implementation of eos_loss
function eos_loss(model, distances::AbstractVector, weights::AbstractVector, fitresult, args...)
    return dot(weights, distances)
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
        softmax!(weights, -convert.(promote_type(Tr, Tf), distances); prefactor=Tf(alpha))
    else
        # Set weights to the uniform distribution
        fill!(weights, Tf(1.0) / T_instances)
    end
    return nothing
end

"""
    calculate_eos_weights(model, fitresult, alpha, args...)

Calculate EOS weights for data (args...) using a fitted model.

# Arguments
- `model`: A fitted MLJ model that implements `eos_distances`
- `fitresult`: The result from fitting the model
- `alpha`: Entropic regularisation parameter
- `args...`: Input data to calculate weights fo

# Returns
- Vector of weights in [0,1] that sum to 1

# See Also
- [`eos_weights`](@ref): Calculate EOS weights from distances using the closed-form solution
  from Horenko (2022).

"""
function calculate_eos_weights(model, fitresult, alpha::Real, args...)
    model_args = MLJModelInterface.reformat(model, args...)
    distances = eos_distances(model, fitresult, model_args...)
    return eos_weights(distances, alpha)
end

"""
    eos_outlier_scores(model, fitresult, alpha, args...)

Calculate outlier scores (exp(-T * weights)) for data (args...) using a fitted model.

Higher scores indicate more outlying samples.

# Arguments
- `model`: A fitted MLJ model that implements `eos_distances`
- `fitresult`: The result from fitting the model
- `alpha`: Entropic regularisation parameter
- `args...`: Input data to score

# Returns
- Vector of outlier scores in (0,1]

# See Also
- [`eos_weights`](@ref): Calculate EOS weights from distances using the closed-form solution
  from Horenko (2022).
- [`calculate_eos_weights`](@ref): Calculate EOS weights for data (args...) using a fitted model.

"""
function eos_outlier_scores(model, fitresult, alpha::Real, args...)
    weights = calculate_eos_weights(model, fitresult, alpha, args...)
    return outlier_scores(weights)
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
- `target_Deff::Real`: The target effective dimension.

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
    target_Deff::Real;
    normalise::Bool=true,
    atol::Real=1e-6,
    kwargs...,
)
    # Validate alpha_range
    alpha_range[1] > 0 && alpha_range[2] > 0 ||
        error("alpha_range must contain positive values")
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
    found_alpha = Roots.find_zero(
        objective, alpha_range, Roots.Chandrapatla(); atol=atol, kwargs...
    )

    # Check convergence
    final_residual = abs(objective(found_alpha))
    if final_residual > atol
        error(
            "Root finding failed to converge. Final residual: $final_residual > $atol. " *
            "Try increasing maxiters or relaxing atol.",
        )
    end

    # Return the final weights and the alpha that produced them
    return (weights=eos_weights(distances, found_alpha), alpha=found_alpha)
end

"""
    calculate_eos_weights(model, fitresult, alpha_range, target_Deff, args...; <kwargs>)

Calculate EOS weights for a model by searching for an `alpha` that yields a target effective
dimension `target_Deff`.

This convenience wrapper first computes `distances` using the provided model, then calls
the corresponding `eos_weights` method to find the `alpha` that matches the `target_Deff`.

# Arguments
- `model`: A fitted MLJ model that implements `eos_distances`.
- `fitresult`: The result from fitting the model.
- `alpha_range::Tuple{<:Real,<:Real}`: The search range for `alpha`.
- `target_Deff::Real`: The target effective dimension for the weights.
- `args...`: Input data.

# Keyword Arguments
- `kwargs...`: Additional keyword arguments forwarded to `eos_weights` (e.g., `normalise`, `atol`, `maxiters`).
  See `?eos_weights` for details.

# Returns
- `(weights, alpha)`: A tuple with the calculated `weights` and the found `alpha`.

# See Also
- [`eos_weights`](@ref): Calculate EOS weights from distances using the closed-form solution
  from Horenko (2022).

"""
function calculate_eos_weights(
    model,
    fitresult,
    alpha_range::Tuple{<:Real,<:Real},
    target_Deff::Real,
    args...;
    kwargs...,
)
    model_args = MLJModelInterface.reformat(model, args...)
    distances = eos_distances(model, fitresult, model_args...)

    return eos_weights(distances, alpha_range, target_Deff; kwargs...)
end

"""
    eos_outlier_scores(model, fitresult, alpha_range, target_Deff, args...; <kwargs>)

Calculate EOS outlier scores by searching for an `alpha` that yields a target effective
dimension `target_Deff`.

This method finds the `alpha` that produces EOS weights with a specific effective
dimension, and then computes the corresponding outlier scores (`exp(-T * weights)`).
It serves as a convenience wrapper around `calculate_eos_weights`.

# Arguments
- `model`: A fitted MLJ model that implements `eos_distances`.
- `fitresult`: The result from fitting the model.
- `alpha_range::Tuple{<:Real,<:Real}`: The search range for `alpha`.
- `target_Deff::Real`: The target effective dimension for the *weights*.
- `args...`: Input data.

# Keyword Arguments
- `kwargs...`: Additional keyword arguments forwarded to `eos_weights` (e.g., `normalise`, `atol`, `maxiters`).
  See `?eos_weights` for details.

# Returns
- `(scores, alpha)`: A tuple with the calculated outlier scores and the found `alpha`.

# See Also
- [`eos_weights`](@ref): Calculate EOS weights from distances using the closed-form solution
  from Horenko (2022).
- [`calculate_eos_weights`](@ref): Calculate EOS weights for data (args...) using a fitted model.

"""
function eos_outlier_scores(
    model,
    fitresult,
    alpha_range::Tuple{<:Real,<:Real},
    target_Deff::Real,
    args...;
    kwargs...,
)
    # Find the appropriate weights and alpha by calling the corresponding eos_weights function.
    result = calculate_eos_weights(model, fitresult, alpha_range, target_Deff, args...; kwargs...)

    # Calculate outlier scores and return with the found alpha.
    return (scores=outlier_scores(result.weights), alpha=result.alpha)
end

# Helper function to get outlier scores from weights
function outlier_scores(weights::AbstractVector{<:Real})
    return exp.(-length(weights) .* weights)
end
