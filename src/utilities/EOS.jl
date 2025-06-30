# TODO: use a more advanced root-finding method
# TODO: use upper and lower bounds from the "waterfall plot"
function ess_calibrate(losses; ρ = 0.8, atol = 1e-10, maxiter = 60)

    ℓ = collect(losses)          # ensure we can index - TODO: unnecessary, just use losses
    T = length(ℓ)
    ess_target = ρ * T

    # if all equal no weighting is possible nor needed - TODO: add tolerance to this check
    iszero(std(losses)) && return (Inf, fill(1/T, T))

    # Similarly, if ess_target = T, then all weights are 1/T
    (ρ >= 1) && return (Inf, fill(1/T, T))
    @assert 0 < ρ < 1 "ρ must be between 0 and 1"

    # helper: weights via log-sum-exp to avoid under-/overflow
    weights(α) = begin
        logw = @. -(ℓ) / α
        m    = maximum(logw)              # log-sum-exp shift
        w    = @. exp(logw - m)
        w ./= sum(w)                      # normalise
    end

    ess(α) = begin
        w = weights(α)
        1 / sum(abs2, w)                  # ESS = 1/∑w²
    end

    # --- bracket a solution --------------------------------------------------
    α_low  = eps(eltype(ℓ))                      # ESS → 1 here
    α_high = maximum(ℓ) - minimum(ℓ) + eps()     # ESS ≈ T here - not guaranteed!

    # TODO: dynamically adjust bounds
    # --- bisection on a *log* scale (monotone ESS) ---------------------------
    for _ in 1:maxiter
        α_mid = √(α_low * α_high)                # geometric mean
        ess_mid = ess(α_mid)

        # stop if close enough (or numeric saturation)
        if !isnan(ess_mid) && abs(ess_mid - ess_target) < atol
            return α_mid, weights(α_mid)
        end

        if isnan(ess_mid) || ess_mid < ess_target
            α_low = α_mid    # need *larger* α to raise ESS
        else
            α_high = α_mid   # need *smaller* α
        end
    end

    α_hat = √(α_low * α_high)                     # fall-back
    return α_hat, weights(α_hat)
end
