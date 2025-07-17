# Initialisation for fitting an EOS model
function initialise(
    eos::EOSWrapper, verbosity::Int, args, T_instances::Int, Tf::Type
)
    # Perform an initial fit with uniform weights
    weights = fill(Tf(1/T_instances), T_instances)
    inner_fitresult, inner_cache, inner_report = MMI.fit(eos.model, verbosity - 1, args...)
    distances = EntropicLearning.eos_distances(eos.model, inner_fitresult, args...)
    EntropicLearning.update_weights!(weights, distances, eos.alpha)

    # Store losses for convergence tracking
    loss = fill(Tf(Inf), eos.max_iter + 1)
    loss[1] = EntropicLearning.eos_loss(eos.model, distances, weights, inner_fitresult, args...) - eos.alpha * EntropicLearning.entropy(weights)

    return weights, distances, loss, inner_fitresult, inner_cache, inner_report
end


# Fit function
function _fit!(
    weights::AbstractVector{Tf},
    distances::AbstractVector{Tf},
    loss::AbstractVector{Tf},
    inner_fitresult,
    inner_cache,
    inner_report,
    eos::EOSWrapper, verbosity::Int, args, to::TimerOutput
) where {Tf<:AbstractFloat}
    # Initialise iteration counter
    iterations = 0

    # --- Main Optimisation Loop ---
    @timeit to "Training" begin
        for iter in 1:eos.max_iter
            # Increment iteration counter
            iterations += 1

            # θ-step: Fit model with current weights
            @timeit to "inner_fit" inner_fitresult, inner_cache, inner_report = MMI.update(
                eos.model, verbosity - 1, inner_fitresult, inner_cache, args..., weights
            )

            # Get distances from fitted model using reformatted data
            @timeit to "distances" distances .= EntropicLearning.eos_distances(
                eos.model, inner_fitresult, args...
            )

            # w-step: Update weights using closed-form solution
            @timeit to "update_weights" EntropicLearning.update_weights!(weights, distances, eos.alpha)

            # Compute objective function for convergence check
            @timeit to "loss" loss[iter + 1] = EntropicLearning.eos_loss(eos.model, distances, weights, inner_fitresult, args...) - eos.alpha * EntropicLearning.entropy(weights)

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

    return inner_fitresult, inner_cache, inner_report, iterations, to
end


# TODO: refactor other components of MMI.fit into separate functions
