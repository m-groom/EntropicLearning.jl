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
using Tables
import ..EntropicLearning

const MMI = MLJModelInterface

export eSPAClassifier

MMI.@mlj_model mutable struct eSPAClassifier <: MMI.Probabilistic
    K::Int = 3::(_ > 0)
    epsC::Float64 = 1e-2::(0.0 <= _ < Inf)
    epsW::Float64 = 1e-1::(_ > 0.0)
    kpp_init::Bool = true::(_ in (true, false))
    mi_init::Bool = true::(_ in (true, false))
    max_iter::Int = 200::(_ > 0)
    tol::Float64 = 1e-8::(_ > 0.0)
    random_state::Union{AbstractRNG,Integer} = Random.default_rng()
end

# Fit Result Structure
struct eSPAFitResult{
    Tm<:AbstractMatrix,Tv<:AbstractVector,Tg<:AbstractMatrix,Tc<:AbstractVector
}
    C::Tm       # Centroids: D x K
    W::Tv       # Feature weights: D-element vector
    L::Tm       # Conditional probabilities for clusters: M x K
    G::Tg       # Cluster affiliations: K × T
    classes::Tc # Vector of unique class labels (CategoricalValue)
end

# Include core eSPA functions for intitialisation, training and prediction
include("core.jl")
include("extras.jl")
include("frontend.jl") # MLJ data front-end

# MLJ Interface
function MMI.fit(
    model::eSPAClassifier,
    verbosity::Int,
    X_mat,
    y_int,
    column_names,
    classes,
    w=nothing,
)
    # Initialise the timer
    to = TimerOutput()

    # Extract dimensions and get Π
    Tf = eltype(X_mat)                                  # Floating point type
    D_features, T_instances = size(X_mat)               # Dimensions
    M_classes = length(classes)                         # Total number of classes
    Pi_mat = get_pi(y_int, M_classes, Tf)               # Target matrix
    classes_seen = MMI.decoder(classes)(unique(y_int))  # Classes seen in training data

    # Ensure weights are normalised
    if !isnothing(w)
        weights = format_weights(w, y_int, Tf)
    else
        weights = fill(Tf(1 / T_instances), T_instances)
    end

    # --- Initialisation ---
    @timeit to "Initialisation" begin
        C, W, L, G = initialise(model, X_mat, y_int, D_features, T_instances, M_classes)
        K_current = size(C, 2)                      # Current number of clusters
        loss = fill(Tf(Inf), model.max_iter + 1)    # Loss for each iteration
        iter = 0                                    # Iteration counter
        loss[1] = calc_loss(X_mat, Pi_mat, C, W, L, G, model.epsC, model.epsW, weights)
    end

    # --- Main Optimisation Loop ---
    @timeit to "Training" begin
        while !converged(loss, iter, model.max_iter, model.tol)
            # Update iteration counter
            iter += 1

            # Evaluation of the Γ-step
            @timeit to "G" update_G!(G, X_mat, Pi_mat, C, W, L, model.epsC, weights)

            # Discard empty boxes
            notEmpty, K_new = find_empty(G)
            if K_new < K_current
                @timeit to "Prune" C, L, G = remove_empty(C, L, G, notEmpty)
                K_current = copy(K_new)
            end

            # Evaluation of the W-step
            @timeit to "W" update_W!(W, X_mat, C, G, model.epsW, weights)

            # Evaluation of the C-step
            @timeit to "C" update_C!(C, X_mat, G, weights)

            # Evaluation of the Λ-step
            @timeit to "L" update_L!(L, Pi_mat, G)

            # Update loss
            @timeit to "Loss" loss[iter + 1] = calc_loss(
                X_mat, Pi_mat, C, W, L, G, model.epsC, model.epsW, weights
            )

            # Check if loss function has increased
            check_loss(loss, iter, verbosity)
        end
    end

    # Warn if the maximum number of iterations was reached
    check_iter(iter, model.max_iter, verbosity)

    # --- Unbiasing step ---
    @timeit to "Unbias" begin
        # Unbias Γ
        update_G!(G, X_mat, Pi_mat, C, W, L, Tf(0.0), weights)

        # Discard empty boxes
        notEmpty, K_new = find_empty(G)
        if K_new < K_current
            C, L, G = remove_empty(C, L, G, notEmpty)
            K_current = copy(K_new)
        end

        # Unbias Λ
        update_L!(L, Pi_mat, G)
    end

    # Estimate the effective number of parameters
    Deff = EntropicLearning.effective_dimension(W)
    n_params = Deff * (K_current + 1) + (M_classes - 1) * K_current

    # --- Return fitresult, cache and report ---
    fitresult = eSPAFitResult(C, W, L, G, classes)
    cache = nothing
    report = (
        iterations=iter,
        loss=loss[1:(iter + 1)],
        timings=to,
        n_params=n_params,
        classes=classes_seen,
        features=column_names,
    )

    return (fitresult, cache, report)
end

function MMI.predict(model::eSPAClassifier, fitresult::eSPAFitResult, X_mat)
    # TODO: store G_new in the report
    Pi_new, G_new = predict_proba(model, fitresult.C, fitresult.W, fitresult.L, X_mat)
    probabilities = transpose(Pi_new)

    return MMI.UnivariateFinite(fitresult.classes, probabilities)
end

function MMI.fitted_params(::eSPAClassifier, fitresult::eSPAFitResult)
    return (C=fitresult.C, W=fitresult.W, L=fitresult.L)
end

function MMI.feature_importances(::eSPAClassifier, fitresult::eSPAFitResult, report)
    W = fitresult.W
    importance = one(eltype(W)) .- exp.(-length(W) .* W)
    # Create pairs of feature_name => importance
    return [report.features[i] => importance[i] for i in eachindex(importance)]
end

# MLJ Traits
MMI.reports_feature_importances(::Type{<:eSPAClassifier}) = true
MMI.iteration_parameter(::Type{<:eSPAClassifier}) = :max_iter
MMI.supports_weights(::Type{<:eSPAClassifier}) = true

MMI.metadata_model(
    eSPAClassifier;
    input_scitype=Union{MMI.Table(MMI.Continuous),AbstractMatrix{<:MMI.Continuous}},
    target_scitype=AbstractVector{<:MMI.Finite},
    human_name="eSPA Classifier",
    load_path="EntropicLearning.eSPA.eSPAClassifier",
)

########## Documentation ##########

"""
$(MMI.doc_header(eSPAClassifier))

entropy-optimal Sparse Probabilistic Approximation (eSPA) classifier, a clustering-based
method designed for classification tasks. The method uses entropic regularisation to
simultaneously learn feature weights and cluster assignments for classification.
This implementation is based on the Python implementation by Davide Bassetti:
https://github.com/davbass/entlearn.

# Training data

In MLJ or MLJBase, bind an instance `model` to data with

    mach = machine(model, X, y)

where

- `X`: any table of input features (e.g., a `DataFrame`) whose columns
  each have `Continuous` element scitype; check column scitypes with `schema(X)`

- `y`: the target, which can be any `AbstractVector` whose element
  scitype is `<:OrderedFactor` or `<:Multiclass`; check the scitype
  with `scitype(y)`

Train the machine with `fit!(mach, rows=...)`.


# Hyperparameters

- `K::Int = 3`: Initial number of clusters to find.

- `epsC::Float64 = 1e-2`: Regularisation parameter for the classification term in the loss
  function.

- `epsW::Float64 = 1e-1`: Regularisation parameter for the entropy of the feature weights.

- `kpp_init::Bool = true`: If `true`, uses k-means++ for centroid initialization. If `false`,
  centroids are initialised by randomly selecting data points.

- `mi_init::Bool = true`: If `true`, feature weights `W` are initialised using the mutual
  information between features and classes. If `false`, they are initialised randomly and
  then normalised.

- `max_iter::Int = 200`: Maximum number of iterations for the main optimisation loop.

- `tol::Float64 = 1e-8`: Tolerance for convergence. The algorithm stops if the relative change
  in loss between iterations is less than `tol`.

- `random_state::Union{AbstractRNG,Integer} = Random.default_rng()`: Seed or AbstractRNG for random number
  generation, ensuring reproducibility. Can be an `Int` or an `AbstractRNG` instance.


# Operations

- `predict(mach, Xnew)`: return probabilistic predictions of the target given
  features `Xnew` having the same scitype as `X` above. Predictions are based on
  learned cluster assignments and conditional probabilities.

- `predict_mode(mach, Xnew)`: instead return the mode (most likely class) of each
  prediction above.


# Fitted parameters

The fields of `fitted_params(mach)` are:

- `C`: Centroids matrix (D × K), where D is the number of features and K is the number of clusters

- `W`: Feature weights vector (D × 1), representing the learned importance of each feature

- `L`: Conditional probabilities matrix (M × K), where M is the number of classes

- `classes`: Vector of unique class labels


# Report

The fields of `report(mach)` are:

- `iterations`: Number of iterations taken by the algorithm

- `loss`: Loss function values at each iteration

- `timings`: Timings for each step of the algorithm

- `G`: Cluster assignment matrix (T × K), where T is the number of instances and K is the number of clusters


# Examples

```
using MLJ

# Load example dataset
X, y = @load_iris

# Create eSPA classifier with custom parameters
eSPA = @load eSPAClassifier pkg=EntropicLearning
model = eSPA(K=3, epsC=1e-3, epsW=1e-1, random_state=101)
mach = machine(model, X, y)
fit!(mach)

# Make predictions
Xnew = (sepal_length = [6.4, 7.2, 7.4],
        sepal_width = [2.8, 3.0, 2.8],
        petal_length = [5.6, 5.8, 6.1],
        petal_width = [2.1, 1.6, 1.9])

yhat = predict(mach, Xnew)     # probabilistic predictions
predict_mode(mach, Xnew)       # point predictions
pdf.(yhat, "virginica")        # probabilities for "virginica" class

# Access fitted parameters
fp = fitted_params(mach)
fp.C                           # learned centroids
fp.W                           # learned feature weights
fp.L                           # conditional probabilities for each cluster
report(mach).G                 # cluster assignment matrix

# Example with different initialization
model2 = eSPA(K=3, kpp_init=false, mi_init=false, random_state=42)
mach2 = machine(model2, X, y)
fit!(mach2)
```

See also the original references:
- [Horenko 2020](https://doi.org/10.1162/neco_a_01296)
- [Vecchi et al. 2022](https://doi.org/10.1162/neco_a_01490)

"""
eSPAClassifier

end # module eSPA
