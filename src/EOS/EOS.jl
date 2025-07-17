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
    atol::Float64
end

mutable struct ProbabilisticEOSWrapper{M} <: MMI.Probabilistic
    model::M
    alpha::Float64
    tol::Float64
    max_iter::Int
    atol::Float64
end

mutable struct UnsupervisedEOSWrapper{M} <: MMI.Unsupervised
    model::M
    alpha::Float64
    tol::Float64
    max_iter::Int
    atol::Float64
end

# Union type for all EOSWrapper types
const EOSWrapper{M} = Union{
    DeterministicEOSWrapper{M},ProbabilisticEOSWrapper{M},UnsupervisedEOSWrapper{M}
} where {M}
Base.parentmodule(::Type{<:EOSWrapper}) = EOS
Base.fieldnames(::Type{<:EOSWrapper}) = (:model, :alpha, :tol, :max_iter, :atol)

# External keyword constructor
function EOSWrapper(
    args...; model=nothing, alpha::Real=1.0, tol::Real=1e-8, max_iter::Integer=100, atol::Real=1e-6
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
            atom, Float64(alpha), Float64(tol), max_iter, Float64(atol)
        )
    elseif atom isa MMI.Probabilistic
        wrapper = ProbabilisticEOSWrapper{typeof(atom)}(
            atom, Float64(alpha), Float64(tol), max_iter, Float64(atol)
        )
    elseif atom isa MMI.Unsupervised
        wrapper = UnsupervisedEOSWrapper{typeof(atom)}(
            atom, Float64(alpha), Float64(tol), max_iter, Float64(atol)
        )
    else
        throw(
            ArgumentError(
                "$(typeof(atom)) does not appear to be a supported MLJ model type"
            ),
        )
    end
    # Check that wrapped model supports weights
    if !MMI.supports_weights(typeof(wrapper.model))
        throw(ArgumentError("Wrapped model type $(typeof(wrapper.model)) must support sample weights."))
    end

    message = MMI.clean!(wrapper)
    isempty(message) || @warn message

    return wrapper
end

# Clean method for parameter validation
function MMI.clean!(wrapper::EOSWrapper)
    err = ""
    if wrapper.alpha <= 0
        err *= "alpha must be positive, got $(wrapper.alpha). Resetting to 1.0."
        wrapper.alpha = 1.0
    end
    if wrapper.tol <= 0
        err *= "tol must be positive, got $(wrapper.tol). Resetting to 1e-8."
        wrapper.tol = 1e-8
    end
    if wrapper.max_iter <= 0
        err *= "max_iter must be positive, got $(wrapper.max_iter). Resetting to 100."
        wrapper.max_iter = 100
    end
    if wrapper.atol <= 0
        err *= "atol must be positive, got $(wrapper.atol). Resetting to 1e-6."
        wrapper.atol = 1e-6
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
    # Inner constructor to ensure ESS is normalised (i.e. between 1/T and 1)
    function EOSFitResult(inner_fitresult, distances, ESS)
        T_train = length(distances)
        if ESS > 1
            ESS /= T_train
        elseif ESS < 1/T_train
            ESS = 1/T_train
        end
        new{typeof(inner_fitresult),typeof(ESS)}(inner_fitresult, distances, ESS)
    end
end

# ==============================================================================
# Fit Methods
# ==============================================================================

include("frontend.jl")  # Data front-end
include("core.jl")

# Common implementation - TODO: separate into fit and update methods
function MMI.fit(eos::EOSWrapper, verbosity::Int, args, T_instances::Int, Tf::Type)
    # Initialise the timer
    to = TimerOutput()

    # TODO: make this a separate function
    # --- Initialisation ---
    @timeit to "Initialisation" begin
        weights = fill(Tf(1/T_instances), T_instances)
        inner_fitresult, inner_cache, inner_report = MMI.fit(eos.model, verbosity - 1, args...)
        distances = EntropicLearning.eos_distances(eos.model, inner_fitresult, args...)
        EntropicLearning.update_weights!(weights, distances, eos.alpha)
        # Store losses for convergence tracking
        loss = fill(Tf(Inf), eos.max_iter + 1)
        iterations = 0
        # loss[1] = eos_loss(eos.model, distances, weights, inner_report, inner_fitresult, inner_cache) - eos.alpha * EntropicLearning.entropy(weights)
        loss[1] =  EntropicLearning.eSPA.calc_loss(
            args[1], args[2], inner_fitresult.C, inner_fitresult.W, inner_fitresult.L, inner_fitresult.G, eos.model.epsC, eos.model.epsW, weights
        ) - eos.alpha * EntropicLearning.entropy(weights)
    end

    # TODO: make this a separate function
    # --- Main Optimisation Loop ---
    @timeit to "Training" begin
        for iter in 1:eos.max_iter
            # Increment iteration counter
            iterations += 1

            loss_before = loss[iter]
            # θ-step: Fit model with current weights
            @timeit to "inner_fit" inner_fitresult, inner_cache, inner_report = MMI.update(
                eos.model, verbosity - 1, inner_fitresult, inner_cache, args..., weights
            )
            # loss_after = eos_loss(eos.model, distances, weights, inner_report, inner_fitresult, inner_cache) - eos.alpha * EntropicLearning.entropy(weights)
            loss_after = EntropicLearning.eSPA.calc_loss(
                args[1], args[2], inner_fitresult.C, inner_fitresult.W, inner_fitresult.L, inner_fitresult.G, eos.model.epsC, eos.model.epsW, weights
            ) - eos.alpha * EntropicLearning.entropy(weights)
            if loss_after > loss_before
                println("θ-step: Loss increased at iteration $iter by $(loss_after - loss_before)")
            end
            loss_before = loss_after

            # Get distances from fitted model using reformatted data
            @timeit to "distances" distances .= EntropicLearning.eos_distances(
                eos.model, inner_fitresult, args...
            )

            # loss_after = eos_loss(eos.model, distances, weights, inner_report, inner_fitresult, inner_cache) - eos.alpha * EntropicLearning.entropy(weights)
            loss_after = EntropicLearning.eSPA.calc_loss(
                args[1], args[2], inner_fitresult.C, inner_fitresult.W, inner_fitresult.L, inner_fitresult.G, eos.model.epsC, eos.model.epsW, weights
            ) - eos.alpha * EntropicLearning.entropy(weights)
            if loss_after > loss_before
                println("distances: Loss increased at iteration $iter by $(loss_after - loss_before)")
            end
            loss_before = loss_after

            # w-step: Update weights using closed-form solution
            @timeit to "update_weights" EntropicLearning.update_weights!(weights, distances, eos.alpha)
            # loss_after = eos_loss(eos.model, distances, weights, inner_report, inner_fitresult, inner_cache) - eos.alpha * EntropicLearning.entropy(weights)
            loss_after = EntropicLearning.eSPA.calc_loss(
                args[1], args[2], inner_fitresult.C, inner_fitresult.W, inner_fitresult.L, inner_fitresult.G, eos.model.epsC, eos.model.epsW, weights
            ) - eos.alpha * EntropicLearning.entropy(weights)
            if loss_after > loss_before
                println("w-step: Loss increased at iteration $iter by $(loss_after - loss_before)")
            end
            loss_before = loss_after

            # Compute objective function for convergence check
            # @timeit to "loss" loss[iter + 1] = eos_loss(eos.model, distances, weights, inner_report, inner_fitresult, inner_cache) - eos.alpha * EntropicLearning.entropy(weights)
            @timeit to "loss" loss[iter + 1] = EntropicLearning.eSPA.calc_loss(
                args[1], args[2], inner_fitresult.C, inner_fitresult.W, inner_fitresult.L, inner_fitresult.G, eos.model.epsC, eos.model.epsW, weights
            ) - eos.alpha * EntropicLearning.entropy(weights)

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
        weights=weights,
        inner_report=inner_report,
    )
    cache = inner_cache

    return (fitresult, cache, report)
end

# ==============================================================================
# Transform and Predict Methods
# ==============================================================================

function MMI.transform(eos::EOSWrapper, fitresult::EOSFitResult, args)
    T_train = length(fitresult.distances)   # Number of training instances
    # Get distances for the new data
    dist = EntropicLearning.eos_distances(eos.model, fitresult.inner_fitresult, args...)
    T_test = length(dist)   # Number of test instances

    # Combine training and test distances
    append!(dist, fitresult.distances)

    # Calculate the range of alpha values to search over to match the training ESS
    amin = min(T_test/T_train, T_train/T_test)
    amax = max(T_test/T_train, T_train/T_test)
    alpha_range = (0.1 * amin * eos.alpha, 10 * amax * eos.alpha)
    ESS = fitresult.ESS

    # Calculate the weights and new alpha
    weights_full, alpha = EntropicLearning.eos_weights(dist, alpha_range, ESS, atol=eos.atol)
    # Convert weights to outlier scores
    scores_full = EntropicLearning.outlier_scores(weights_full)

    # Extract the weights and scores for the test data
    weights_new = weights_full[1:T_test]
    scores_new = scores_full[1:T_test]
    return scores_new, (weights=weights_new, alpha=alpha)
end

# For supervised models only
function MMI.predict(
    eos::Union{DeterministicEOSWrapper,ProbabilisticEOSWrapper},
    fitresult::EOSFitResult,
    args,
)
    # Get predictions from the inner model
    result = MMI.predict(eos.model, fitresult.inner_fitresult, args...)
    # Get outlier scores for the new data
    scores, report = MMI.transform(eos, fitresult, args)
    # Return the predictions and outlier scores
    if :predict in MMI.reporting_operations(typeof(eos.model))
        return result[1], (scores=scores, report..., inner_report=result[2])
    else
        return result, (scores=scores, report...)
    end
end

# ==============================================================================
# Fitted Parameters and Feature Importances
# ==============================================================================

function MMI.fitted_params(eos::EOSWrapper, fitresult::EOSFitResult)
    return (
        weights=EntropicLearning.eos_weights(fitresult.distances, eos.alpha),
        inner_params=MMI.fitted_params(eos.model, fitresult.inner_fitresult),
    )
end

function MMI.feature_importances(eos::EOSWrapper, fitresult::EOSFitResult, report)
    return MMI.feature_importances(eos.model, fitresult.inner_fitresult, report)
end

function MMI.training_losses(::EOSWrapper, report)
    return report.loss
end

# ==============================================================================
# Documentation
# ==============================================================================

"""
$(MMI.doc_header(EOSWrapper))

Entropic Outlier Sparsification (EOS) wrapper for MLJ models.

This meta-algorithm wraps any MLJ model that supports sample weights and provides
outlier detection capabilities through entropic regularisation of the sample weights.

# Training data

In MLJ or MLJBase, bind an instance `eos` to data with

    mach = machine(eos, X, y)

where

- `X`: any table of input features (e.g., a `DataFrame`) whose columns
  each have a scitype compatible with the wrapped model `eos.model`; check column scitypes with `schema(X)` and model-compatible scitypes with `input_scitype(eos.model)`

- `y`: the target, which can be any `AbstractVector` whose element
  scitype is compatible with the wrapped model `eos.model`; check the scitype
  with `scitype(y)` and model-compatible scitypes with `target_scitype(eos.model)`

Train the machine with `fit!(mach, rows=...)`.


# Hyperparameters

- `model::M`: The wrapped MLJ model (must support sample weights)

- `alpha::Float64 = 1.0`: Entropic regularisation parameter (>0). Larger values lead to more uniform weights.

- `tol::Float64 = 1e-8`: Convergence tolerance for the iterative algorithm

- `max_iter::Int = 100`: Maximum number of iterations

- `atol::Float64 = 1e-6`: Absolute tolerance for root finding during predict/transform operations

# Operations

- `transform(mach, Xnew)`: return outlier scores for the new data `Xnew` having the same scitype as `X` above.

- `predict(mach, Xnew)`: return predictions of the target from the wrapped modelgiven
  features `Xnew` having the same scitype as `X` above.

# Fitted parameters

The fields of `fitted_params(mach)` are:

- `weights`: The learned sample weights

- `inner_params`: The fitted parameters of the wrapped model


# Report

The fields of `report(mach)` are:

- `iterations`: Number of iterations taken by the algorithm

- `loss`: Loss function values at each iteration

- `timings`: Timings for each step of the algorithm

- `ESS`: Effective sample size of the weights

- `inner_report`: The report of the wrapped model


# Examples

```
using MLJ

# Load example dataset
X, y = @load_iris

# Create eSPA classifier
eSPA = @load eSPAClassifier pkg=EntropicLearning
model = eSPA(K=3, epsC=1e-3, epsW=1e-1, random_state=101)

# Wrap the model with EOS
EOS = @load EOSWrapper pkg=EntropicLearning
eos = EOS(model=model, alpha=0.1)
mach = machine(eos, X, y)
fit!(mach)

# Make predictions
Xnew = (sepal_length = [6.4, 7.2, 7.4],
        sepal_width = [2.8, 3.0, 2.8],
        petal_length = [5.6, 5.8, 6.1],
        petal_width = [2.1, 1.6, 1.9])

yhat = predict(mach, Xnew)     # probabilistic predictions
predict_mode(mach, Xnew)       # point predictions
pdf.(yhat, "virginica")        # probabilities for "virginica" class

# Get outlier scores for the new data
scores = transform(mach, Xnew)

# Access fitted parameters
fp = fitted_params(mach)
fp.weights                     # learned sample weights for the training data
fp.inner_params                # fitted parameters of the wrapped model
```

See also the original references:
- [Horenko 2022](https://doi.org/10.1073/pnas.2119659119)

"""
EOSWrapper

end # module EOS
