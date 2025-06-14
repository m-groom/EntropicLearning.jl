module eSPA

using MLJModelInterface
using StatsBase: sample
using LinearAlgebra
using Random
using SparseArrays
using Clustering: initseeds!, KmppAlg, copyseeds!
using Clustering.Distances: SqEuclidean, WeightedSqEuclidean
using TimerOutputs
using NearestNeighbors: KDTree, knn, inrange, Chebyshev
using SpecialFunctions: digamma
using Statistics: mean, std

# Include common functions
include("../common/functions.jl")

const MMI = MLJModelInterface

export eSPAClassifier

"""
    eSPAClassifier <: MLJModelInterface.Probabilistic

entropy-optimal Sparse Probabilistic Approximation(eSPA) classifier.

This model implements the eSPA algorithm, a clustering-based method designed for
classification tasks. Key references:
    - https://doi.org/10.1162/neco_a_01296
    - https://doi.org/10.1162/neco_a_01490

Fields:
- `K::Int`: Initial number of clusters to find. Default: `10`.
- `epsC::Float64`: Regularization parameter for the classification term in the loss
  function. Default: `1e-3`.
- `epsW::Float64`: Regularization parameter for the entropy of feature weights. Encourages
  smoother/sparser feature weights. Default: `1e-3`.
- `kpp_init::Bool`: If `true`, uses k-means++ for centroid initialization. If `false`,
  centroids are initialized by randomly selecting data points. Default: `true`.
- `mi_init::Bool`: If `true`, feature weights `W_` are initialized using the mutural
  information between features and classes. If `false`, they are initialized randomly and
  then normalized. Default: `true`.
- `iterative_pred::Bool`: If `true`, performs iterative refinement of assignments during
  the prediction phase. Default: `false`.
- `unbias::Bool`: If `true`, performs an unbiasing step after the main optimization loop to
  recalculate cluster assignments without the influence of `epsC`. Default: `false`.
- `max_iter::Int`: Maximum number of iterations for the main optimization loop. Default: `200`.
- `tol::Float64`: Tolerance for convergence. The algorithm stops if the relative change
  in loss between iterations is less than `tol`. Default: `1e-8`.
- `random_state::Union{AbstractRNG,Integer}`: Seed or AbstractRNG for random number
  generation, ensuring reproducibility. Can be an `Int` or an `AbstractRNG` instance.
  Default: `GLOBAL_RNG`.
"""
MMI.@mlj_model mutable struct eSPAClassifier <: MMI.Probabilistic
    K::Int = 3::(_ > 0)
    epsC::Float64 = 1e-3::(_ >= 0)
    epsW::Float64 = 1e-1::(_ > 0)
    kpp_init::Bool = true::(_ in (true, false))
    mi_init::Bool = true::(_ in (true, false))
    iterative_pred::Bool = false::(_ in (true, false))
    unbias::Bool = false::(_ in (true, false))
    max_iter::Int = 200::(_ > 0)
    tol::Float64 = 1e-8::(_ > 0)
    random_state::Union{AbstractRNG,Integer} = Random.GLOBAL_RNG
end

# Fit Result Structure
struct eSPAFitResult{Tm<:AbstractMatrix,Tv<:AbstractVector,Tc<:AbstractVector}
    C::Tm       # Centroids: D x K
    W::Tv       # Feature weights: D-element vector
    L::Tm       # Conditional probabilities for clusters: M x K
    classes::Tc # Vector of unique class labels (CategoricalValue)
end

# Include core eSPA functions for intitialisation, training and prediction
include("core.jl")
include("extras.jl")

# MLJ Interface
function MMI.fit(model::eSPAClassifier, verbosity::Int, X, y)
    # Initialise the random number generator and timer
    rng = _get_rng(model.random_state)
    to = TimerOutput()

    # TODO: write a data front-end
    X_mat = MMI.matrix(X; transpose=true)
    D_features, T_instances = size(X_mat)
    classes = MMI.classes(y[1])
    M_classes = length(classes)
    y_int = MMI.int(y)

    # TODO: make this a function
    Tf = eltype(X_mat)
    Pi_mat = zeros(Tf, M_classes, T_instances)
    if T_instances > 0
        for t in 1:T_instances
            Pi_mat[y_int[t], t] = one(Tf)
        end
    end

    # --- Initialization ---
    @timeit to "Initialisation" begin
        C, W, L, G = initialise(
            model, X_mat, y_int, D_features, T_instances, M_classes; rng=rng
        )
        K_current = size(C, 2)                  # Current number of clusters
        loss = fill(Inf, model.max_iter + 1)    # Loss for each iteration
        iter = 0                                # Iteration counter
        loss[1] = calc_loss(X_mat, Pi_mat, C, W, L, G, model.epsC, model.epsW)
    end

    # --- Main Optimization Loop ---
    @timeit to "Training" begin
        while (iter == 0 || abs((loss[iter + 1] - loss[iter]) / loss[iter]) > model.tol) &&
            iter < model.max_iter
            # Update iteration counter
            iter += 1

            # Evaluation of the Γ-step
            @timeit to "G" update_G!(G, X_mat, Pi_mat, C, W, L, model.epsC)

            # Discard empty boxes
            notEmpty, K_new = find_empty(G)
            if K_new < K_current
                @timeit to "Prune" C, L, G = remove_empty(C, L, G, notEmpty)
                K_current = copy(K_new)
            end

            # Evaluation of the W-step
            @timeit to "W" update_W!(W, X_mat, C, G, model.epsW)

            # Evaluation of the C-step
            @timeit to "C" update_C!(C, X_mat, G)

            # Evaluation of the Λ-step
            @timeit to "L" update_L!(L, Pi_mat, G)

            # Update loss
            @timeit to "Loss" loss[iter + 1] = calc_loss(
                X_mat, Pi_mat, C, W, L, G, model.epsC, model.epsW
            )

            # Check if loss function has increased
            if loss[iter + 1] - loss[iter] > eps(Tf) && verbosity > 0
                @warn "Loss function has increased at iteration $iter by $(loss[iter + 1] - loss[iter])"
            end
        end
    end

    # Warn if the maximum number of iterations was reached
    if iter >= model.max_iter && verbosity > 0
        @warn "Maximum number of iterations reached"
    end

    # --- Unbiasing step ---
    @timeit to "Unbias" begin
        if model.unbias
            # Unbias Γ
            update_G!(G, X_mat, Pi_mat, C, W, L, Tf(0.0))

            # if model.iterative_pred
            #     # TODO: implement iterative prediction
            # end

            # Discard empty boxes
            notEmpty, K_new = find_empty(G)
            if K_new < K_current
                C, L, G = remove_empty(C, L, G, notEmpty)
                K_current = copy(K_new)
            end

            # Unbias Λ - TODO: this should only be done if iterative_pred is false
            update_L!(L, Pi_mat, G)
        end
    end

    # --- Return fitresult, cache and report ---
    fitresult = eSPAFitResult(C, W, L, classes)
    cache = nothing
    report = (iterations=iter, loss=loss[1:(iter + 1)], timings=to, G=G)

    return (fitresult, cache, report)
end

function MMI.predict(model::eSPAClassifier, fitresult::eSPAFitResult, Xnew)

    # TODO: write a data front-end
    X_mat = MMI.matrix(Xnew; transpose=true)

    # Get dimensions
    T_instances = size(X_mat, 2)
    M_classes = length(fitresult.classes)

    Pi_new = _predict_proba(model, fitresult, X_mat)
    @assert size(Pi_new) == (M_classes, T_instances)
    probabilities = collect(Pi_new')

    if size(probabilities, 1) == 0 || size(probabilities, 2) == 0
        return MMI.UnivariateFinite[]
    else
        return MMI.UnivariateFinite(fitresult.classes, probabilities)
    end
end

function MMI.fitted_params(::eSPAClassifier, fitresult::eSPAFitResult)
    return (C=fitresult.C, W=fitresult.W, L=fitresult.L)
end

# MLJ Traits
MMI.reports_feature_importances(::Type{<:eSPAClassifier}) = true
MMI.iteration_parameter(::Type{<:eSPAClassifier}) = :max_iter

MMI.metadata_model(
    eSPAClassifier;
    input_scitype=Table(Continuous),
    target_scitype=AbstractVector{<:Finite},
    human_name="eSPA Classifier",
    load_path="EntropicLearning.eSPA.eSPAClassifier",
)

end # module eSPA
