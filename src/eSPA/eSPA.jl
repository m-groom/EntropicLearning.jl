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
    random_state::Union{AbstractRNG,Integer} = Random.default_rng()
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
    # Initialise the timer
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
            model, X_mat, y_int, D_features, T_instances, M_classes
        )
        K_current = size(C, 2)                  # Current number of clusters
        loss = fill(Tf(Inf), model.max_iter + 1)    # Loss for each iteration
        iter = 0                                # Iteration counter
        loss[1] = calc_loss(X_mat, Pi_mat, C, W, L, G, model.epsC, model.epsW)
    end

    # --- Main Optimisation Loop ---
    @timeit to "Training" begin
        while !converged(loss, iter, model.max_iter, model.tol)
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
            check_loss(loss, iter, verbosity)
        end
    end

    # Warn if the maximum number of iterations was reached
    check_iter(iter, model.max_iter, verbosity)

    # --- Unbiasing step ---
    @timeit to "Unbias" begin
        if model.unbias
            # Unbias Γ
            update_G!(G, X_mat, Pi_mat, C, W, L, Tf(0.0))

            if model.iterative_pred
                P = Matrix{Tf}(undef, M_classes, T_instances)
                update_P!(P, L, G)
                iterative_predict!(P, G, model, X_mat, C, W, L; verbosity=verbosity)
            end

            # Discard empty boxes
            notEmpty, K_new = find_empty(G)
            if K_new < K_current
                C, L, G = remove_empty(C, L, G, notEmpty)
                K_current = copy(K_new)
            end

            if !model.iterative_pred
                # Unbias Λ
                update_L!(L, Pi_mat, G)
            end
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

    Pi_new = predict_proba(model, fitresult, X_mat)
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

function MMI.feature_importances(::eSPAClassifier, fitresult::eSPAFitResult, report)
    # TODO: store feature names in the report

    # Extract feature weights from fitresult
    W = fitresult.W

    # Create feature names (since they're not stored in fitresult)
    feature_names = [Symbol("feature_$i") for i in eachindex(W)]

    # Create pairs of feature_name => importance
    return [feature_names[i] => W[i] for i in eachindex(W)]
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

- `epsC::Float64 = 1e-3`: Regularisation parameter for the classification term in the loss
  function.

- `epsW::Float64 = 1e-1`: Regularisation parameter for the entropy of the feature weights.

- `kpp_init::Bool = true`: If `true`, uses k-means++ for centroid initialization. If `false`,
  centroids are initialised by randomly selecting data points.

- `mi_init::Bool = true`: If `true`, feature weights `W` are initialised using the mutual
  information between features and classes. If `false`, they are initialised randomly and
  then normalised.

- `iterative_pred::Bool = false`: If `true`, performs iterative refinement of cluster assignments
  during the prediction phase.

- `unbias::Bool = false`: If `true`, performs an unbiasing step after the main optimisation loop to
  recalculate cluster assignments without the influence of `epsC`.

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
