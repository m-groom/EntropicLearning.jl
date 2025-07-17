# Helper function to get the loss from the inner model - TODO: add documentation in case users need to override this
function eos_loss(model, distances::AbstractVector, weights::AbstractVector, fitresult, args...)
    return dot(weights, distances)
end
