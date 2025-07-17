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


# TODO: implement _fit function


# TODO: refactor other components of MMI.fit into separate functions
