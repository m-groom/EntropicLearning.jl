module EOS

using MLJModelInterface
using LinearAlgebra
using SparseArrays
using TimerOutputs
import ..EntropicLearning

# Include common functions - TODO: call from EntropicLearning instead
# include("../common/functions.jl")

# Include EOS utility functions - TODO: call from EntropicLearning instead
# include("../utilities/eos.jl")

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

# Input/output scitypes - inherit from wrapped model
MMI.input_scitype(::Type{<:EOSWrapper{M}}) where {M} = MMI.input_scitype(M)
MMI.target_scitype(::Type{<:DeterministicEOSWrapper{M}}) where {M} = MMI.target_scitype(M)
MMI.target_scitype(::Type{<:ProbabilisticEOSWrapper{M}}) where {M} = MMI.target_scitype(M)

# ==============================================================================
# Fit Result Structure
# ==============================================================================

struct EOSFitResult{F,T<:AbstractFloat}
    inner_fitresult::F
    weights::AbstractVector{T}  # TODO: store distances instead
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

# Common implementation
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

    # Initialise weights and distances - TODO: fit the model without weights first
    Tf = Float64
    weights = fill(Tf(1/T_instances), T_instances)
    distances = zeros(Tf, T_instances)
    inner_fitresult = nothing
    inner_cache = nothing
    inner_report = nothing

    # Store losses for convergence tracking
    loss = fill(Tf(Inf), eos.max_iter)
    iterations = 0

    # Prepare arguments for fitting the inner model
    fit_args = (args..., weights)

    # --- Main Optimisation Loop ---
    @timeit to "Training" begin
        for iter in 1:eos.max_iter
            # Increment iteration counter
            iterations += 1

            # θ-step: Fit model with current weights - TODO: use update if it is available
            @timeit to "inner_fit" inner_fitresult, inner_cache, inner_report = MMI.fit(
                eos.model, verbosity - 1, fit_args...
            )

            # Get distances from fitted model using reformatted data - TODO: use mutating version if it is available
            @timeit to "distances" distances .= EntropicLearning.eos_distances(
                eos.model, inner_fitresult, args...
            )

            # w-step: Update weights using closed-form solution
            @timeit to "update_weights" EntropicLearning.update_weights!(weights, distances, eos.alpha)

            # Compute objective function for convergence check
            @timeit to "loss" loss[iter] =
                dot(weights, distances) - eos.alpha * EntropicLearning.entropy(weights)

            # Check if loss function has increased
            if iter > 1 && loss[iter] - loss[iter - 1] > eps(Tf)
                verbosity > 0 &&
                    @warn "Loss function has increased at iteration $iter by $(loss[iter] - loss[iter-1])"
            end

            # Check convergence
            if iter > 1 && abs((loss[iter] - loss[iter - 1]) / loss[iter]) <= eos.tol
                break
            end
        end
    end

    if iterations >= eos.max_iter && verbosity > 0
        @warn "EOS reached maximum iterations without converging"
    end

    # --- Return fitresult, cache and report ---
    fitresult = EOSFitResult(inner_fitresult, weights)
    report = (
        iterations=iterations,
        loss=loss[1:iterations],
        timings=to,
        ESS=EntropicLearning.effective_dimension(weights),
        inner_report=inner_report,
    )
    cache = inner_cache

    return (fitresult, cache, report)
end

# ==============================================================================
# Transform and Predict Methods
# ==============================================================================

# Transform always returns weights (for both supervised and unsupervised)
# TODO: modify so that weights also include distances from training data
function MMI.transform(eos::EOSWrapper, fitresult::EOSFitResult, Xnew)
    # Reformat new data for the wrapped model
    args = MMI.reformat(eos.model, Xnew) # Assume first argument is the data matrix
    # TODO: call root-finding method instead?
    return EntropicLearning.calculate_eos_weights(
        eos.model, fitresult.inner_fitresult, eos.alpha, args[1]
    )
end

# For supervised models only - TODO: return new weights in report (from transform)
function MMI.predict(
    eos::Union{DeterministicEOSWrapper,ProbabilisticEOSWrapper},
    fitresult::EOSFitResult,
    Xnew,
)
    # Reformat new data for the wrapped model and pass through
    args = MMI.reformat(eos.model, Xnew)
    return MMI.predict(eos.model, fitresult.inner_fitresult, args...)
end

# ==============================================================================
# Fitted Parameters and Feature Importances
# ==============================================================================

function MMI.fitted_params(eos::EOSWrapper, fitresult::EOSFitResult)
    return (
        alpha=fitresult.alpha,
        weights=fitresult.weights,
        inner_fitted_params=MMI.fitted_params(eos.model, fitresult.inner_fitresult),
    )
end

function MMI.feature_importances(eos::EOSWrapper, fitresult::EOSFitResult, report)
    return MMI.feature_importances(eos.model, fitresult.inner_fitresult, report)
end

end # module EOS
