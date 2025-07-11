module EOS

using MLJModelInterface
using LinearAlgebra
using SparseArrays
using TimerOutputs
import ..EntropicLearning

const MMI = MLJModelInterface

export EOSWrapper

# ==============================================================================
# EOSWrapper Model Definition
# ==============================================================================

# Define wrapper structs for each MLJ model type
mutable struct DeterministicEOSWrapper{M} <: MMI.Deterministic
    model::M
    alpha::Float64
    tol::Float64
    max_iter::Int
end

mutable struct ProbabilisticEOSWrapper{M} <: MMI.Probabilistic
    model::M
    alpha::Float64
    tol::Float64
    max_iter::Int
end

mutable struct UnsupervisedEOSWrapper{M} <: MMI.Unsupervised
    model::M
    alpha::Float64
    tol::Float64
    max_iter::Int
end

# TODO: make MLJ-compliant docstring
"""
    EOSWrapper{M}

Entropic Outlier Sparsification (EOS) wrapper for MLJ models.

This meta-algorithm wraps any MLJ model that supports sample weights and provides
outlier detection capabilities through entropic regularisation of the sample weights.

# Parameters
- `model::M`: The wrapped MLJ model (must support sample weights)
- `alpha::Float64`: Entropic regularisation parameter (>0). Larger values lead to more uniform weights.
- `tol::Float64`: Convergence tolerance for the iterative algorithm
- `max_iter::Int`: Maximum number of iterations

# Example
```julia
# Wrap a classifier with EOS
using MLJ, EntropicLearning

# Load a model that supports weights
Tree = @load DecisionTreeClassifier pkg=DecisionTree

# Create EOS-wrapped model
eos_model = EOSWrapper(Tree(), alpha=1.0)

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
const EOSWrapper{M} = Union{
    DeterministicEOSWrapper{M},ProbabilisticEOSWrapper{M},UnsupervisedEOSWrapper{M}
} where {M}
Base.parentmodule(::Type{<:EOSWrapper}) = EOS

# External keyword constructor
function EOSWrapper(
    args...; model=nothing, alpha::Real=1.0, tol::Real=1e-6, max_iter::Integer=100
)
    length(args) < 2 || throw(ArgumentError("Too many positional arguments"))

    if length(args) === 1
        atom = first(args)
        model === nothing ||
            @warn "Using `model=$atom`. Ignoring specification " * "`model=$model`. "
    else
        model === nothing &&
            throw(ArgumentError("model parameter is required for EOSWrapper"))
        atom = model
    end

    # Create appropriate wrapper type based on wrapped model type
    if atom isa MMI.Deterministic
        wrapper = DeterministicEOSWrapper{typeof(atom)}(
            atom, Float64(alpha), Float64(tol), max_iter
        )
    elseif atom isa MMI.Probabilistic
        wrapper = ProbabilisticEOSWrapper{typeof(atom)}(
            atom, Float64(alpha), Float64(tol), max_iter
        )
    elseif atom isa MMI.Unsupervised
        wrapper = UnsupervisedEOSWrapper{typeof(atom)}(
            atom, Float64(alpha), Float64(tol), max_iter
        )
    else
        throw(
            ArgumentError(
                "$(typeof(atom)) does not appear to be a supported MLJ model type"
            ),
        )
    end

    message = MMI.clean!(wrapper)
    isempty(message) || throw(ArgumentError(message))
    return wrapper
end

# Clean method for parameter validation
function MMI.clean!(wrapper::EOSWrapper)
    err = ""
    if wrapper.alpha <= 0
        err *= "alpha must be positive, got $(wrapper.alpha). "
    end
    if wrapper.tol <= 0
        err *= "tol must be positive, got $(wrapper.tol). "
    end
    if wrapper.max_iter <= 0
        err *= "max_iter must be positive, got $(wrapper.max_iter). "
    end
    # Check that wrapped model supports weights
    if !MMI.supports_weights(typeof(wrapper.model))
        err *= "Wrapped model type $(typeof(wrapper.model)) must support sample weights. "
    end
    return err
end

# MLJ traits
MMI.iteration_parameter(::Type{<:EOSWrapper}) = :max_iter
MMI.constructor(::Type{<:EOSWrapper}) = EOSWrapper
MMI.metadata_model(
    EOSWrapper;
    human_name="Entropic Outlier Sparsification",
    load_path="EntropicLearning.EOS.EOSWrapper",
)
function MMI.reports_feature_importances(::Type{<:EOSWrapper{M}}) where {M}
    MMI.reports_feature_importances(M)
end
MMI.is_pure_julia(::Type{<:EOSWrapper{M}}) where {M} = MMI.is_pure_julia(M)
MMI.supports_training_losses(::Type{<:EOSWrapper}) = true
MMI.reporting_operations(::Type{<:EOSWrapper}) = (:predict,:transform)

# Input/output scitypes - inherit from wrapped model
MMI.input_scitype(::Type{<:EOSWrapper{M}}) where {M} = MMI.input_scitype(M)
MMI.target_scitype(::Type{<:DeterministicEOSWrapper{M}}) where {M} = MMI.target_scitype(M)
MMI.target_scitype(::Type{<:ProbabilisticEOSWrapper{M}}) where {M} = MMI.target_scitype(M)

# ==============================================================================
# Fit Result Structure
# ==============================================================================

struct EOSFitResult{F,T<:AbstractFloat}
    inner_fitresult::F
    distances::AbstractVector{T}
    ESS::T
end

# ==============================================================================
# Fit Methods
# ==============================================================================

# Unsupervised case
function MMI.fit(eos::UnsupervisedEOSWrapper, verbosity::Int, X)
    return _fit(eos, verbosity, X, nothing)
end

# Supervised case (works for both Deterministic and Probabilistic)
function MMI.fit(eos::EOSWrapper, verbosity::Int, X, y)
    # This will match DeterministicEOSWrapper and ProbabilisticEOSWrapper
    return _fit(eos, verbosity, X, y)
end

# Common implementation - TODO: separate into fit and update methods
function _fit(eos::EOSWrapper, verbosity::Int, X, y=nothing)
    # Initialise the timer
    to = TimerOutput()

    # Reformat data for the wrapped model
    T_instances = MMI.nrows(X)
    if isnothing(y)
        args = MMI.reformat(eos.model, X)
    else
        args = MMI.reformat(eos.model, X, y)
    end

    # --- Initialisation ---
    @timeit to "Initialisation" begin
        Tf = eltype(X[1])
        weights = fill(Tf(1/T_instances), T_instances)
        inner_fitresult, inner_cache, inner_report = MMI.fit(eos.model, verbosity - 1, args...)
        distances = EntropicLearning.eos_distances(eos.model, inner_fitresult, args...)
        EntropicLearning.update_weights!(weights, distances, eos.alpha)
        # Store losses for convergence tracking
        loss = fill(Tf(Inf), eos.max_iter + 1)
        iterations = 0
        loss[1] = dot(weights, distances) - eos.alpha * EntropicLearning.entropy(weights)
    end

    # --- Main Optimisation Loop ---
    @timeit to "Training" begin
        for iter in 1:eos.max_iter
            # Increment iteration counter
            iterations += 1

            # θ-step: Fit model with current weights
            @timeit to "inner_fit" inner_fitresult, inner_cache, inner_report = MMI.update(
                eos.model, verbosity - 1, inner_fitresult, inner_cache, args..., weights
            )

            # Get distances from fitted model using reformatted data - TODO: use mutating version if it is available
            @timeit to "distances" distances .= EntropicLearning.eos_distances(
                eos.model, inner_fitresult, args...
            )

            # w-step: Update weights using closed-form solution
            @timeit to "update_weights" EntropicLearning.update_weights!(weights, distances, eos.alpha)

            # Compute objective function for convergence check
            @timeit to "loss" loss[iter + 1] = eos_loss(eos.model, distances, weights, inner_report, inner_fitresult, inner_cache) - eos.alpha * EntropicLearning.entropy(weights)

            # Check if loss function has increased
            if loss[iter + 1] - loss[iter] > eps(Tf)
                verbosity > 0 &&
                    @warn "Loss function has increased at iteration $iter by $(loss[iter + 1] - loss[iter])"
            end

            # Check convergence
            if abs((loss[iter + 1] - loss[iter]) / loss[iter]) <= eos.tol
                break
            end
        end
    end

    if iterations >= eos.max_iter && verbosity > 0
        @warn "EOS reached maximum iterations without converging"
    end

    # --- Return fitresult, cache and report ---
    fitresult = EOSFitResult(inner_fitresult, distances, EntropicLearning.effective_dimension(weights, normalise=true))
    report = (
        iterations=iterations,
        loss=loss[1:iterations + 1],
        timings=to,
        ESS=EntropicLearning.effective_dimension(weights),
        inner_report=inner_report,
    )
    cache = inner_cache

    return (fitresult, cache, report)
end

# Helper function to get the loss from the inner model - TODO: add documentation in case users need to override this
function eos_loss(model, distances::AbstractVector, weights::AbstractVector, report, args...)
    if MMI.supports_training_losses(typeof(model))
        l = MMI.training_losses(model, report)[end]
    else
        l = dot(weights, distances)
    end
    return l
end

# ==============================================================================
# Transform and Predict Methods
# ==============================================================================
# TODO: write a data front-end so that transform (and predict) get the model-specific data format
# Transform always returns weights (for both supervised and unsupervised)
function MMI.transform(eos::EOSWrapper, fitresult::EOSFitResult, Xnew)
    args = MMI.reformat(eos.model, Xnew)
    dist = EntropicLearning.eos_distances(eos.model, fitresult.inner_fitresult, args...)
    append!(dist, fitresult.distances)
    amin = min(MMI.nrows(Xnew)/length(fitresult.distances), length(fitresult.distances)/MMI.nrows(Xnew))
    amax = max(MMI.nrows(Xnew)/length(fitresult.distances), length(fitresult.distances)/MMI.nrows(Xnew))
    alpha_range = (amin * eos.alpha, amax * eos.alpha + 1e-5)
    ESS = fitresult.ESS
    weights_full, alpha = EntropicLearning.eos_weights(dist, alpha_range, ESS, atol=1e-6)
    weights_new = weights_full[1:MMI.nrows(Xnew)]
    return weights_new, (alpha=alpha,)
end

# For supervised models only
function MMI.predict(
    eos::Union{DeterministicEOSWrapper,ProbabilisticEOSWrapper},
    fitresult::EOSFitResult,
    Xnew,
)
    # Reformat new data for the wrapped model and pass through
    args = MMI.reformat(eos.model, Xnew)
    result = MMI.predict(eos.model, fitresult.inner_fitresult, args...)
    weights, report = MMI.transform(eos, fitresult, Xnew)
    if :predict in MMI.reporting_operations(typeof(eos.model))
        return result[1], (weights=weights, inner_report_pred=result[2], report...)
    else
        return result, (weights=weights, report...)
    end
end

# ==============================================================================
# Fitted Parameters and Feature Importances
# ==============================================================================

function MMI.fitted_params(eos::EOSWrapper, fitresult::EOSFitResult)
    return (
        weights=EntropicLearning.eos_weights(fitresult.distances, eos.alpha),
        inner_fitted_params=MMI.fitted_params(eos.model, fitresult.inner_fitresult),
    )
end

function MMI.feature_importances(eos::EOSWrapper, fitresult::EOSFitResult, report)
    return MMI.feature_importances(eos.model, fitresult.inner_fitresult, report)
end

function MMI.training_losses(::EOSWrapper, report)
    return report.loss
end

end # module EOS
