module eSPAClassifier

using MLJModelInterface
# using Tables
using StatsBase: sample
using LinearAlgebra
using Random
# using Distributions: pdf
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

export eSPA

"""
    eSPA <: MLJModelInterface.Probabilistic

An Entropic Semisupervised Positive Unlabeled (eSPA) classifier.

This model implements the eSPA algorithm, a clustering-based approach primarily designed for
Positive-Unlabeled (PU) learning but adaptable for standard classification. It iteratively
refines cluster centroids, feature weights, and class posteriors for clusters to optimize a
loss function combining reconstruction error, classification accuracy, and feature weight
entropy.

Fields:
- `K::Int`: Initial number of clusters to find.
- `epsC::Float64`: Regularization parameter for the classification term in the loss
  function.
  Controls the influence of class labels on cluster assignments.
- `epsW::Float64`: Regularization parameter for the entropy of feature weights. Encourages
  smoother/sparser feature weights.
- `kpp_init::Bool`: If `true`, uses k-means++ for centroid initialization. If `false`,
  centroids are initialized by randomly selecting data points.
- `iterative_pred::Bool`: If `true`, performs iterative refinement of assignments during
  the prediction phase (`predict_proba`).
- `unbias::Bool`: If `true`, performs an unbiasing step after the main optimization loop to
  refine class posteriors without the influence of `epsC`.
- `mi_init::Bool`: If `true`, feature weights `W_` are initialized using the mutural
  information between features and classes. If `false`, they are initialized randomly and
  then normalized.
- `max_iter::Int`: Maximum number of iterations for the main optimization loop.
- `tol::Float64`: Tolerance for convergence. The algorithm stops if the absolute change
  in loss between iterations is less than `tol`.
- `random_state::Any`: Seed or AbstractRNG for random number generation, ensuring
  reproducibility. Can be an `Int` or an `AbstractRNG` instance.
- `verbose::Bool`: If `true`, prints progress information during fitting, including
  iteration number, loss, and convergence status. Also enables printing of `TimerOutputs`.
- `debug_loss::Bool`: If `true`, performs additional checks after each substep of the
  algorithm to ensure the loss function does not increase (which is a theoretical property
  of eSPA). Warnings are printed if an increase is detected.
"""
mutable struct eSPA <: MMI.Probabilistic  # TODO: use @mlj_model macro instead
    K::Int
    epsC::Float64
    epsW::Float64
    kpp_init::Bool
    iterative_pred::Bool
    unbias::Bool
    mi_init::Bool
    max_iter::Int
    tol::Float64
    random_state::Any # Can be Int or AbstractRNG
    verbose::Bool
    debug_loss::Bool
end

# Include core eSPA functions (after struct definition)
include("core.jl")

# Include extras
include("extras.jl")

"""
    eSPA(; K=10, epsC=1e-3, epsW=1e-3, kpp_init=false, iterative_pred=false,
         unbias=false, mi_init=true, max_iter=200, tol=1e-8, random_state=123,
         verbose=false, debug_loss=false)

Constructor for the `eSPA` model.

# Arguments
- `K::Int`: Initial number of clusters. Default: `10`.
- `epsC::Float64`: Classification regularization parameter. Default: `1e-3`.
- `epsW::Float64`: Feature weight entropy regularization parameter. Default: `1e-3`.
- `kpp_init::Bool`: If `true` (default), use k-means++ for centroid initialization. If
  `false`, initialize centroids by randomly selecting data points.
- `iterative_pred::Bool`: Whether to use iterative refinement during prediction.
  Default: `false`.
- `unbias::Bool`: Whether to perform an unbiasing step after fitting. Default: `false`.
- `mi_init::Bool`: If `true` (default), initialize feature weights using the mutural
  information between features and classes. If `false`, initialize randomly and normalize.
- `max_iter::Int`: Maximum optimization iterations. Default: `200`.
- `tol::Float64`: Convergence tolerance for the loss function. Default: `1e-8`.
- `random_state::Any`: Seed or RNG for reproducibility. Default: `123`.
- `verbose::Bool`: Enable verbose output during fitting. Default: `false`.
- `debug_loss::Bool`: Enable loss increase checks after each substep. Default: `false`.
"""
function eSPA(;
    K::Int=10,
    epsC::Float64=1e-3,
    epsW::Float64=1e-3,
    kpp_init::Bool=true,
    iterative_pred::Bool=false,
    unbias::Bool=false,
    mi_init::Bool=false,
    max_iter::Int=200,
    tol::Float64=1e-8,
    random_state=123, # Default seed
    verbose::Bool=false,
    debug_loss::Bool=false,
)
    return eSPA(
        K,
        epsC,
        epsW,
        kpp_init,
        iterative_pred,
        unbias,
        mi_init,
        max_iter,
        tol,
        random_state,
        verbose,
        debug_loss,
    )
end

# Fit Result Structure
struct eSPAFitResult{CMatType,WVecType,LMatType,ClassVecType}
    C_::CMatType         # Centroids: D x K_final
    W_::WVecType         # Feature weights: D-element vector
    L_::LMatType         # Class posteriors for clusters: M x K_final
    K_final::Int         # Final number of clusters
    classes_::ClassVecType # Vector of unique class labels (CategoricalValue)
    epsC::Float64        # Store epsC for iterative prediction
    # model_params for iterative pred: max_iter, tol, verbose
    max_iter::Int
    tol::Float64
    verbose::Bool
end

# Helper to get RNG
_get_rng(random_state::Int) = MersenneTwister(random_state)
_get_rng(random_state::AbstractRNG) = random_state

# Helper function for debug loss checks
function _debug_check_loss_substep(
    loss_before::Float64,
    loss_after::Float64,
    step_name::String,
    iteration::Int,
    tolerance_threshold::Float64,
)
    if loss_after > loss_before + tolerance_threshold
        println(
            "Warning: Loss increase after $step_name at iter $iteration: $loss_before -> $loss_after (diff: $(loss_after - loss_before))",
        )
    end
end

# MLJ Interface
function MMI.fit(model::eSPA, verbosity::Int, X, y)
    rng = _get_rng(model.random_state)
    to = TimerOutput()

    X_mat = MMI.matrix(X; transpose=true)
    D_features, T_instances = size(X_mat)

    classes = MMI.classes(y[1])
    M_classes = length(classes)
    y_int = MMI.int(y)

    Pi_mat = zeros(eltype(X_mat), M_classes, T_instances)
    if T_instances > 0
        for t in 1:T_instances
            Pi_mat[y_int[t], t] = one(eltype(X_mat))
        end
    end

    # --- Initialization ---
    C, W, L, G = initialise(
        model, X_mat, Pi_mat, D_features, T_instances, M_classes; rng=rng
    )

    K_current_ref = Ref(K_current_val)

    losses = zeros(Float64, model.max_iter)
    # Store initial loss if loop will run and loss is valid
    if model.max_iter > 0 && initial_loss != Inf
        losses[1] = initial_loss
    # Ensure losses[1] is Inf if not otherwise set and loop runs
    elseif model.max_iter > 0
        losses[1] = Inf
    end

    iterations_run = 0
    debug_tol_threshold = model.tol * 10.0

    # --- Main Optimization Loop ---
    for i in 1:model.max_iter
        if K_current_ref[] == 0
            # Correctly set iterations if loop breaks due to K=0
            iterations_run = i - 1
            break
        end

        loss_at_iter_start = if model.debug_loss && K_current_ref[] > 0
            @timeit to "calc_loss" calc_loss(
                X_mat,
                Pi_mat,
                C_,
                W_,
                L_,
                G_,
                model.epsC,
                model.epsW,
                K_current_ref[],
                T_instances,
            )
        else
            0.0 # Dummy value, not used if debug_loss is false
        end

        # 1. Update G and Remove Empty Clusters
        @timeit to "update_G!" begin
            G_ = update_G!(
                X_mat,
                C_,
                W_,
                L_,
                Pi_mat,
                model.epsC,
                K_current_ref[],
                T_instances,
                W_metric,
            )
        end
        @timeit to "remove_empty" begin
            C_, L_, G_ = remove_empty(C_, L_, G_, K_current_ref)
        end
        if model.debug_loss && K_current_ref[] > 0
            loss_after_G = @timeit to "calc_loss" calc_loss(
                X_mat,
                Pi_mat,
                C_,
                W_,
                L_,
                G_,
                model.epsC,
                model.epsW,
                K_current_ref[],
                T_instances,
            )
            _debug_check_loss_substep(
                loss_at_iter_start,
                loss_after_G,
                "G update/remove_empty",
                i,
                debug_tol_threshold,
            )
            loss_at_iter_start = loss_after_G # Update baseline for next substep
        end

        # Early exit condition
        if K_current_ref[] == 0 || (K_current_ref[] == 1 && i > 1)
            if model.verbose && K_current_ref[] <= 1 && i > 1
                println("Early exit: K = $(K_current_ref[]), iter $i")
            end
            iterations_run = i
            if K_current_ref[] > 0
                losses[i] = @timeit to "calc_loss" calc_loss(
                    X_mat,
                    Pi_mat,
                    C_,
                    W_,
                    L_,
                    G_,
                    model.epsC,
                    model.epsW,
                    K_current_ref[],
                    T_instances,
                )
            else
                losses[i] = Inf
            end
            break
        end

        # 2. Update W
        @timeit to "update_W!" begin
            update_W!(W_, X_mat, C_, G_, model.epsW, K_current_ref[], T_instances)
            # CRITICAL: Update W_metric immediately after W_ changes
            W_metric = WeightedSqEuclidean(W_)
        end
        if model.debug_loss && K_current_ref[] > 0
            loss_after_W = @timeit to "calc_loss" calc_loss(
                X_mat,
                Pi_mat,
                C_,
                W_,
                L_,
                G_,
                model.epsC,
                model.epsW,
                K_current_ref[],
                T_instances,
            )
            _debug_check_loss_substep(
                loss_at_iter_start, loss_after_W, "W update", i, debug_tol_threshold
            )
            loss_at_iter_start = loss_after_W
        end

        # 3. Update C
        @timeit to "update_C!" begin
            update_C!(C_, X_mat, G_, K_current_ref[])
        end
        if model.debug_loss && K_current_ref[] > 0
            loss_after_C = @timeit to "calc_loss" calc_loss(
                X_mat,
                Pi_mat,
                C_,
                W_,
                L_,
                G_,
                model.epsC,
                model.epsW,
                K_current_ref[],
                T_instances,
            )
            _debug_check_loss_substep(
                loss_at_iter_start, loss_after_C, "C update", i, debug_tol_threshold
            )
            loss_at_iter_start = loss_after_C
        end

        # 4. Update L
        @timeit to "update_L!" begin
            update_L!(L_, Pi_mat, G_, K_current_ref[], M_classes)
        end
        if model.debug_loss && K_current_ref[] > 0
            loss_after_L = @timeit to "calc_loss" calc_loss(
                X_mat,
                Pi_mat,
                C_,
                W_,
                L_,
                G_,
                model.epsC,
                model.epsW,
                K_current_ref[],
                T_instances,
            )
            _debug_check_loss_substep(
                loss_at_iter_start, loss_after_L, "L update", i, debug_tol_threshold
            )
        end

        # Calculate and store overall loss for this iteration
        current_iter_loss = @timeit to "calc_loss" calc_loss(
            X_mat,
            Pi_mat,
            C_,
            W_,
            L_,
            G_,
            model.epsC,
            model.epsW,
            K_current_ref[],
            T_instances,
        )
        losses[i] = current_iter_loss

        # Verbose output for overall loss change
        if model.verbose && verbosity >= 2 && i > 1
            # Check for significant increase from previous iteration's final loss
            if losses[i - 1] - current_iter_loss < -model.tol
                println(
                    "Warning: Overall loss increase at iter $i: $(losses[i-1]) -> $(current_iter_loss)",
                )
            end
        end

        # Convergence Check
        converged_this_iter = false
        if i > 1
            if abs(losses[i - 1] - current_iter_loss) < model.tol
                converged_this_iter = true
            end
            # If K=0, initial_loss might be Inf. Convergence can occur if K becomes 0 during
            # loop.
        elseif i == 1 && (
            current_iter_loss == -Inf ||
            (K_current_ref[] == 0 && initial_loss == Inf) ||
            (K_current_ref[] == 0 && T_instances == 0)
        )
            converged_this_iter = true
        end

        if converged_this_iter
            if model.verbose && verbosity >= 1
                println("Converged at iter $i, loss: $(current_iter_loss)")
            end
            iterations_run = i
            break
        end
        iterations_run = i # Update iterations_run at the end of a successful iteration
    end

    # --- Unbiasing step ---
    if model.unbias && K_current_ref[] > 0 && T_instances > 0
        if model.verbose && verbosity >= 1
            println("Performing unbiasing step...")
        end
        temp_epsC = 0.0 # Use for G computation in unbiasing

        loss_before_unbias = 0.0
        if model.debug_loss
            loss_before_unbias = @timeit to "calc_loss" calc_loss(
                X_mat,
                Pi_mat,
                C_,
                W_,
                L_,
                G_,
                model.epsC,
                model.epsW,
                K_current_ref[],
                T_instances,
            )
        end

        # Unbias G (W_metric here uses the W_ from the end of the main loop)
        @timeit to "update_G!" begin
            G_ = update_G!(
                X_mat, C_, W_, L_, Pi_mat, temp_epsC, K_current_ref[], T_instances, W_metric
            )
        end
        if model.debug_loss
            loss_after_unbias_G = @timeit to "calc_loss" calc_loss(
                X_mat,
                Pi_mat,
                C_,
                W_,
                L_,
                G_,
                model.epsC,
                model.epsW,
                K_current_ref[],
                T_instances,
            )
            _debug_check_loss_substep(
                loss_before_unbias,
                loss_after_unbias_G,
                "G update (unbias)",
                iterations_run + 1,
                debug_tol_threshold,
            )
            loss_before_unbias = loss_after_unbias_G # Update baseline
        end

        # Unbias L
        @timeit to "update_L!" begin
            update_L!(L_, Pi_mat, G_, K_current_ref[], M_classes)
        end
        if model.debug_loss
            loss_after_unbias_L = @timeit to "calc_loss" calc_loss(
                X_mat,
                Pi_mat,
                C_,
                W_,
                L_,
                G_,
                model.epsC,
                model.epsW,
                K_current_ref[],
                T_instances,
            )
            _debug_check_loss_substep(
                loss_before_unbias,
                loss_after_unbias_L,
                "L update (unbias)",
                iterations_run + 1,
                debug_tol_threshold,
            )
        end
    end

    # --- Prepare report ---
    final_losses_to_report = Float64[]
    if iterations_run > 0
        final_losses_to_report = losses[1:iterations_run]
    elseif model.max_iter == 0 # Loop didn't run because max_iter was 0
        if initial_loss != Inf
            final_losses_to_report = [initial_loss]
        end
        # Add other edge cases if iterations_run is 0 but initial_loss was computed
        # (e.g. K=0 from start)
    elseif iterations_run == 0 &&
        initial_loss != Inf &&
        (model.debug_loss || model.max_iter > 0) # model.max_iter condition from _initialize
        final_losses_to_report = [initial_loss]
    end

    fitresult = eSPAFitResult(
        C_,
        W_,
        L_,
        K_current_ref[],
        classes,
        model.epsC,
        model.max_iter,
        model.tol,
        model.verbose,
    )
    cache = nothing
    report = (
        iterations=iterations_run,
        final_K=K_current_ref[],
        losses=final_losses_to_report,
        timings=to,
    )

    if model.verbose && verbosity >= 1
        println(to)
    end

    return (fitresult, cache, report)
end

# TODO: move this to core.jl
function _predict_proba_internal(
    X_new_mat_T::AbstractMatrix{Float64},
    fr::eSPAFitResult,
    model::eSPA,
    to_predict::TimerOutput,
)
    @timeit to_predict "_predict_proba_internal" begin
        D_features, T_new_samples = size(X_new_mat_T)

        C_ = fr.C_
        W_ = fr.W_
        L_ = fr.L_
        K_final = fr.K_final

        if K_final == 0 || T_new_samples == 0
            M_classes = size(L_, 1)
            num_actual_classes =
                isempty(fr.classes_) ? (M_classes > 0 ? M_classes : 1) : length(fr.classes_)
            if num_actual_classes == 0
                num_actual_classes = 1
            end
            return fill(1.0 / num_actual_classes, num_actual_classes, T_new_samples)
        end

        W_metric_pred = WeightedSqEuclidean(W_)
        Dist_term_G_new = pairwise(W_metric_pred, C_, X_new_mat_T)

        G_new = spzeros(Bool, K_final, T_new_samples)
        if T_new_samples > 0 && K_final > 0
            G_new = sparse(
                ones(Int, T_new_samples), 1:T_new_samples, true, K_final, T_new_samples
            )
            assign_closest!(G_new, Dist_term_G_new)
        end

        slog_L = safelog.(L_)
        Pi_pred_scores = -1.0 .* (slog_L * G_new)
        Pi_pred_iter = zeros(Float64, size(L_, 1), T_new_samples)
        if size(Pi_pred_iter, 2) > 0
            assign_closest!(Pi_pred_iter, Pi_pred_scores)
        end

        if model.iterative_pred
            iter_max_iter = fr.max_iter
            iter_tol = fr.tol
            iter_verbose = fr.verbose

            for iter_idx in 1:iter_max_iter
                G_old_structure = findnz(G_new)

                Dist_term_G_test = pairwise(W_metric_pred, C_, X_new_mat_T)
                slog_L_T = safelog.(L_)'
                Class_term_G_test = fr.epsC .* (slog_L_T * Pi_pred_iter)
                G_values_test = Dist_term_G_test .- Class_term_G_test

                if T_new_samples > 0 && K_final > 0
                    G_new_updated = sparse(
                        ones(Int, T_new_samples),
                        1:T_new_samples,
                        true,
                        K_final,
                        T_new_samples,
                    )
                    assign_closest!(G_new_updated, G_values_test)
                    G_new = G_new_updated
                else
                    G_new = sparse(Int[], Int[], Bool[], K_final, T_new_samples)
                end

                Pi_pred_scores = -1.0 .* (slog_L * G_new)
                if size(Pi_pred_iter, 2) > 0
                    assign_closest!(Pi_pred_iter, Pi_pred_scores)
                end

                G_new_structure = findnz(G_new)
                if G_new_structure[1] == G_old_structure[1] &&
                    G_new_structure[2] == G_old_structure[2]
                    # Check verbosity before printing for iterative predict
                    # Using model.verbose directly, as verbosity argument is for fit's main
                    # control
                    if model.verbose && iter_verbose && verbosity >= 2
                        println(
                            "\tIterative predict_proba converged in $iter_idx iterations."
                        )
                    end
                    break
                end

                if model.verbose &&
                    iter_verbose &&
                    verbosity >= 2 &&
                    iter_idx == iter_max_iter
                    println("\tIterative predict_proba reached max_iter.")
                end
            end
        end

        probabilities = L_ * G_new
        return probabilities
    end
end

function MMI.predict(model::eSPA, fitresult::eSPAFitResult, Xnew)
    to_predict = TimerOutput()
    # Pass model.verbose to internal predict for its own verbose flags if needed
    # The verbosity parameter here is from MLJ, usually for controlling MLJ's own messages.
    @timeit to_predict "predict_proba" begin
        Xnew_mat_T = MMI.matrix(Xnew)'
        probabilities_T = _predict_proba_internal(Xnew_mat_T, fitresult, model, to_predict)
        probabilities = collect(probabilities_T')

        if model.verbose && verbosity >= 2 # Higher MLJ verbosity to show predict timings
            println(to_predict)
        end

        if size(probabilities, 2) != length(fitresult.classes_) &&
            !isempty(fitresult.classes_)
            if isempty(fitresult.classes_) && size(probabilities, 2) > 0
                error(
                    "Cannot create UnivariateFinite with no classes defined in fitresult."
                )
            elseif isempty(fitresult.classes_) && size(probabilities, 2) == 0
                return UnivariateFinite[]
            end
        end

        if size(probabilities, 1) == 0
            return UnivariateFinite[]
        end

        return [
            UnivariateFinite(fitresult.classes_, probs) for probs in eachrow(probabilities)
        ]
    end
end

function MMI.predict_mode(model::eSPA, fitresult::eSPAFitResult, Xnew)
    distributions = MMI.predict_proba(model, fitresult, Xnew)
    if isempty(distributions)
        if !isempty(fitresult.classes_)
            return similar(fitresult.classes_, 0)
        else
            return []
        end
    end
    return MMI.mode.(distributions)
end

# MLJ Traits
MMI.input_scitype(::Type{<:eSPA}) = Table(Continuous)
MMI.target_scitype(::Type{<:eSPA}) = AbstractVector{<:Finite}
MMI.load_path(::Type{<:eSPA}) = "EntropicLearning.eSPA.eSPA"
MMI.supports_weights(::Type{<:eSPA}) = false
function MMI.docstring(::Type{<:eSPA})
    "Julia implementation of eSPA classifier, inspired by entlearn Python package. Uses Clustering.kmeanspp, Distances.jl, sparse G matrix, and TimerOutputs."
end
MMI.human_name(::Type{<:eSPA}) = "eSPA Classifier"
MMI.package_name(::Type{<:eSPA}) = "EntropicLearning"
MMI.package_uuid(::Type{<:eSPA}) = "dummy-uuid-for-eSPA"
MMI.package_url(::Type{<:eSPA}) = "https://github.com/yourusername/EntropicLearning"
MMI.is_pure_julia(::Type{<:eSPA}) = true
MMI.iteration_parameter(::Type{<:eSPA}) = :max_iter

end # module eSPAClassifier
