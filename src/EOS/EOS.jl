module EOS

using MLJModelInterface
using LinearAlgebra

# Include common functions
include("../common/functions.jl")

# Include EOS utility functions
include("../utilities/eos.jl")

const MMI = MLJModelInterface

export EOSDetector

# ==============================================================================
# EOSDetector Model Definition
# ==============================================================================

"""
    EOSDetector{M,S}

Entropic Outlier Sparsification (EOS) wrapper for MLJ models.

This meta-algorithm wraps any MLJ model that supports sample weights and provides
outlier detection capabilities through entropic regularisation of the sample weights.

# Parameters
- `model::M`: The wrapped MLJ model (must support sample weights)
- `α::Float64`: Entropic regularization parameter (>0). Larger values lead to more uniform weights.
- `tol::Float64`: Convergence tolerance for the iterative algorithm
- `max_iter::Int`: Maximum number of iterations

# Example
```julia
# Wrap a classifier with EOS
using MLJ, EntropicLearning

# Load a model that supports weights
Tree = @load DecisionTreeClassifier pkg=DecisionTree

# Create EOS-wrapped model
eos_model = EOSDetector(Tree(), α=1.0)

# Train as usual
mach = machine(eos_model, X, y) |> fit!

# Get predictions
ŷ = predict(mach, Xnew)

# Get outlier scores
outlier_scores = transform(mach, Xnew)
```

# References
Horenko, I. (2022). "Cheap robust learning of data anomalies with analytically
solvable entropic outlier sparsification." PNAS 119(9), e2119659119.
"""
mutable struct EOSDetector{M,S} <: MMI.Model{S}
    model::M
    α::Float64
    tol::Float64
    max_iter::Int

    function EOSDetector(model::M; α=1.0, tol=1e-6, max_iter=100) where M
        α > 0 || error("α must be positive")
        tol > 0 || error("tol must be positive")
        max_iter > 0 || error("max_iter must be positive")

        # Determine if wrapped model is supervised or unsupervised
        S = MMI.is_supervised(M) ? MMI.Supervised() : MMI.Unsupervised()
        return new{M,typeof(S)}(model, α, tol, max_iter)
    end
end

# MLJ model traits
MMI.is_wrapper(::Type{<:EOSDetector}) = true
MMI.supports_weights(::Type{<:EOSDetector{M,S}}) where {M,S} = MMI.supports_weights(M)
MMI.package_name(::Type{<:EOSDetector}) = "EntropicLearning"
MMI.load_path(::Type{<:EOSDetector}) = "EntropicLearning.EOS.EOSDetector"

# Input/output scitypes - inherit from wrapped model
MMI.input_scitype(::Type{<:EOSDetector{M,S}}) where {M,S} = MMI.input_scitype(M)
MMI.target_scitype(::Type{<:EOSDetector{M,MMI.Supervised}}) where M = MMI.target_scitype(M)

# ==============================================================================
# Fit Result Structure
# ==============================================================================

struct EOSFitResult{F,T}
    inner_fitresult::F
    final_weights::Vector{T}
    α::T
    n_iter::Int
end

# ==============================================================================
# Fit Methods
# ==============================================================================

# Unsupervised case
function MMI.fit(eos::EOSDetector{M,MMI.Unsupervised}, verbosity::Int, X) where M
    return _eos_fit(eos, verbosity, X, nothing)
end

# Supervised case
function MMI.fit(eos::EOSDetector{M,MMI.Supervised}, verbosity::Int, X, y) where M
    return _eos_fit(eos, verbosity, X, y)
end

# Common implementation
function _eos_fit(eos::EOSDetector{M,S}, verbosity::Int, X, y) where {M,S}
    # Check that wrapped model supports weights
    if !MMI.supports_weights(M)
        error("Wrapped model type $M must support sample weights. " *
              "Check MLJModelInterface.supports_weights($M)")
    end

    # Check that model supports EOS
    if !supports_eos(M)
        error("Model type $M must implement eos_distances to be EOS-compatible. " *
              "See documentation for implementing eos_distances($M, fitresult, X, [y])")
    end

    n = MMI.nrows(X)

    # Initialize uniform weights
    weights = fill(1/n, n)

    # Storage for convergence tracking
    losses = Float64[]
    inner_fitresult = nothing

    verbosity > 0 && @info "Starting EOS iterations with α=$(eos.α)"

    for iter in 1:eos.max_iter
        # θ-step: Fit model with current weights
        fit_args = isnothing(y) ? (X,) : (X, y)
        inner_fitresult, _, _ = MMI.fit(eos.model, verbosity-1, fit_args...;
                                        weights=weights)

        # Get distances from fitted model
        distances = eos_distances(eos.model, inner_fitresult, X, y)

        # w-step: Update weights using closed-form solution
        new_weights = eos_weights(distances, eos.α)

        # Compute objective function for convergence check
        entropy_term = -sum(w * log(w + eps()) for w in new_weights)
        objective = dot(new_weights, distances) - eos.α * entropy_term
        push!(losses, objective)

        # Check convergence
        if iter > 1 && abs(losses[iter] - losses[iter-1]) < eos.tol
            verbosity > 0 && @info "EOS converged after $iter iterations"
            weights = new_weights
            break
        end

        weights = new_weights

        if verbosity > 1
            @info "EOS iteration $iter: objective = $(losses[iter])"
        end
    end

    if length(losses) == eos.max_iter && verbosity > 0
        @warn "EOS reached maximum iterations without converging"
    end

    fitresult = EOSFitResult(
        inner_fitresult,
        weights,
        eos.α,
        length(losses)
    )

    report = (
        iterations = length(losses),
        convergence_history = losses,
        final_weights = weights
    )

    cache = nothing

    return fitresult, cache, report
end

# ==============================================================================
# Transform and Predict Methods
# ==============================================================================

# Transform always returns outlier scores (for both supervised and unsupervised)
function MMI.transform(eos::EOSDetector, fitresult::EOSFitResult, Xnew)
    distances = eos_distances(eos.model, fitresult.inner_fitresult, Xnew, nothing)
    weights = eos_weights(distances, fitresult.α)
    return 1 .- weights  # Convert to outlier scores
end

# For supervised models, also provide predict
function MMI.predict(eos::EOSDetector{M,MMI.Supervised}, fitresult::EOSFitResult, Xnew) where M
    # Pass through to wrapped model
    return MMI.predict(eos.model, fitresult.inner_fitresult, Xnew)
end

# ==============================================================================
# Fitted Parameters
# ==============================================================================

function MMI.fitted_params(eos::EOSDetector, fitresult::EOSFitResult)
    return (
        α = fitresult.α,
        final_weights = fitresult.final_weights,
        n_iter = fitresult.n_iter,
        inner_fitted_params = MMI.fitted_params(eos.model, fitresult.inner_fitresult)
    )
end


end # module EOS
