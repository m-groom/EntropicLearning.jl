# # TODO: use a more advanced root-finding method
# # TODO: use upper and lower bounds from the "waterfall plot"
# function ess_calibrate(losses; ρ = 0.8, atol = 1e-10, maxiter = 60)

#     ℓ = collect(losses)          # ensure we can index - TODO: unnecessary, just use losses
#     T = length(ℓ)
#     ess_target = ρ * T

#     # if all equal no weighting is possible nor needed - TODO: add tolerance to this check
#     iszero(std(losses)) && return (Inf, fill(1/T, T))

#     # Similarly, if ess_target = T, then all weights are 1/T
#     (ρ >= 1) && return (Inf, fill(1/T, T))
#     @assert 0 < ρ < 1 "ρ must be between 0 and 1"

#     # helper: weights via log-sum-exp to avoid under-/overflow
#     weights(α) = begin
#         logw = @. -(ℓ) / α
#         m    = maximum(logw)              # log-sum-exp shift
#         w    = @. exp(logw - m)
#         w ./= sum(w)                      # normalise
#     end

#     ess(α) = begin
#         w = weights(α)
#         1 / sum(abs2, w)                  # ESS = 1/∑w²
#     end

#     # --- bracket a solution --------------------------------------------------
#     α_low  = eps(eltype(ℓ))                      # ESS → 1 here
#     α_high = maximum(ℓ) - minimum(ℓ) + eps()     # ESS ≈ T here - not guaranteed!

#     # TODO: dynamically adjust bounds
#     # --- bisection on a *log* scale (monotone ESS) ---------------------------
#     for _ in 1:maxiter
#         α_mid = √(α_low * α_high)                # geometric mean
#         ess_mid = ess(α_mid)

#         # stop if close enough (or numeric saturation)
#         if !isnan(ess_mid) && abs(ess_mid - ess_target) < atol
#             return α_mid, weights(α_mid)
#         end

#         if isnan(ess_mid) || ess_mid < ess_target
#             α_low = α_mid    # need *larger* α to raise ESS
#         else
#             α_high = α_mid   # need *smaller* α
#         end
#     end

#     α_hat = √(α_low * α_high)                     # fall-back
#     return α_hat, weights(α_hat)
# end


module EOS

using MLJModelInterface
using LinearAlgebra: dot
using Statistics: mean

const MMI = MLJModelInterface

export EOS, eos_weights, eos_outlier_scores, calculate_eos_weights, eos_distances, supports_eos

# ==============================================================================
# EOS Model Definition
# ==============================================================================

"""
    EOS{M,S}

Entropic Outlier Sparsification (EOS) wrapper for MLJ models.

This meta-algorithm wraps any MLJ model that supports sample weights and provides
outlier detection capabilities through entropic regularization of sample weights.

# Parameters
- `model::M`: The wrapped MLJ model (must support sample weights)
- `α::Float64`: Entropic regularization parameter (>0). Larger values lead to more uniform weights.
- `tol::Float64`: Convergence tolerance for the iterative algorithm
- `max_iter::Int`: Maximum number of iterations

# References
Horenko, I. (2022). "Cheap robust learning of data anomalies with analytically 
solvable entropic outlier sparsification." PNAS 119(9), e2119659119.
"""
mutable struct EOS{M,S} <: MMI.Model{S}
    model::M
    α::Float64
    tol::Float64
    max_iter::Int
    
    function EOS(model::M; α=1.0, tol=1e-6, max_iter=100) where M
        α > 0 || error("α must be positive")
        tol > 0 || error("tol must be positive")
        max_iter > 0 || error("max_iter must be positive")
        
        # Determine if wrapped model is supervised or unsupervised
        S = MMI.is_supervised(M) ? MMI.Supervised() : MMI.Unsupervised()
        return new{M,typeof(S)}(model, α, tol, max_iter)
    end
end

# MLJ model traits
MMI.is_wrapper(::Type{<:EOS}) = true
MMI.supports_weights(::Type{<:EOS{M,S}}) where {M,S} = MMI.supports_weights(M)
MMI.package_name(::Type{<:EOS}) = "EntropicLearning"
MMI.load_path(::Type{<:EOS}) = "EntropicLearning.EOS.EOS"

# Input/output scitypes - inherit from wrapped model
MMI.input_scitype(::Type{<:EOS{M,S}}) where {M,S} = MMI.input_scitype(M)
MMI.target_scitype(::Type{<:EOS{M,MMI.Supervised}}) where M = MMI.target_scitype(M)

# ==============================================================================
# Distance Function Protocol
# ==============================================================================

"""
    eos_distances(model, fitresult, X, [y])

Compute distances/losses for each sample in X using the fitted model.

This function must be implemented for any model to be EOS-compatible.
For supervised models, y may be provided for computing supervised losses.
"""
function eos_distances end

# Default error message
eos_distances(model, args...) = 
    error("Model type $(typeof(model)) must implement eos_distances to be EOS-compatible")

"""
    supports_eos(::Type{ModelType})

Check if a model type supports EOS by implementing the required interface.
"""
supports_eos(::Type) = false
# supports_eos(::Type{<:eSPA}) = true  # Uncomment when eSPA implements eos_distances

# ==============================================================================
# Core EOS Functions
# ==============================================================================

"""
    eos_weights(distances, α)

Calculate EOS weights from distances using the closed-form solution.
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

Calculate EOS weights for data X using a fitted model (mode 2: single-step).
"""
function calculate_eos_weights(model, fitresult, X, α::Real; y=nothing)
    distances = eos_distances(model, fitresult, X, y)
    return eos_weights(distances, α)
end

"""
    eos_outlier_scores(model, fitresult, X, α; y=nothing)

Calculate outlier scores (1 - weight) for data X using a fitted model.
Higher scores indicate more outlying samples.
"""
function eos_outlier_scores(model, fitresult, X, α::Real; y=nothing)
    weights = calculate_eos_weights(model, fitresult, X, α; y=y)
    return 1 .- weights
end

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
function MMI.fit(eos::EOS{M,MMI.Unsupervised}, verbosity::Int, X) where M
    return _eos_fit(eos, verbosity, X, nothing)
end

# Supervised case
function MMI.fit(eos::EOS{M,MMI.Supervised}, verbosity::Int, X, y) where M
    return _eos_fit(eos, verbosity, X, y)
end

# Common implementation
function _eos_fit(eos::EOS{M,S}, verbosity::Int, X, y) where {M,S}
    # Check that wrapped model supports weights
    if !MMI.supports_weights(M)
        error("Wrapped model type $M must support sample weights. " *
              "Check MLJModelInterface.supports_weights($M)")
    end
    
    # Check that model supports EOS
    if !supports_eos(M)
        error("Model type $M must implement eos_distances to be EOS-compatible. " *
              "See documentation for details.")
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
function MMI.transform(eos::EOS, fitresult::EOSFitResult, Xnew)
    distances = eos_distances(eos.model, fitresult.inner_fitresult, Xnew, nothing)
    weights = eos_weights(distances, fitresult.α)
    return 1 .- weights  # Convert to outlier scores
end

# For supervised models, also provide predict
function MMI.predict(eos::EOS{M,MMI.Supervised}, fitresult::EOSFitResult, Xnew) where M
    # Pass through to wrapped model
    return MMI.predict(eos.model, fitresult.inner_fitresult, Xnew)
end

# ==============================================================================
# Fitted Parameters
# ==============================================================================

function MMI.fitted_params(eos::EOS, fitresult::EOSFitResult)
    return (
        α = fitresult.α,
        final_weights = fitresult.final_weights,
        n_iter = fitresult.n_iter,
        inner_fitted_params = MMI.fitted_params(eos.model, fitresult.inner_fitresult)
    )
end

# ==============================================================================
# OutlierDetectionInterface Support
# ==============================================================================

# Check if OutlierDetectionInterface is available
const _ODI_AVAILABLE = try
    @eval using OutlierDetectionInterface
    true
catch
    false
end

if _ODI_AVAILABLE
    # All EOS models are outlier detectors
    OutlierDetectionInterface.is_outlier_detector(::Type{<:EOS}) = true
end

end # module EOS
