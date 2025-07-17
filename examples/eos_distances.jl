using MLJ
using EntropicLearning
using Random
using Statistics
using MLJGLMInterface

# Implementation of eos_distancesfor LinearRegressor from MLJGLMInterface
function EntropicLearning.eos_distances(
    model::MLJGLMInterface.LinearRegressor, fitresult, X, y=nothing
)
    # Convert X to matrix
    Xmat = MLJ.matrix(X)

    # Get coefficients from FitResult
    coefs = fitresult.coefs

    # Get predictions
    if model.fit_intercept
        intercept = coefs[1]  # Intercept
        β = coefs[2:end]  # Feature coefficients
        ŷ = Xmat * β .+ intercept
    else
        ŷ = Xmat * coefs
    end

    if isnothing(y)
        # For transform: use squared distance from prediction to mean of predictions
        ŷ_mean = mean(ŷ)
        distances = (ŷ .- ŷ_mean) .^ 2
    else
        # For fit: use squared residuals
        y_vec = vec(y)  # Ensure y is a vector
        distances = (y_vec .- ŷ) .^ 2
    end

    return distances
end

# Generate synthetic regression data with outliers
X, y = make_regression(
    200,
    5;               # 200 samples, 5 features
    noise=0.5,             # Moderate noise
    sparse=0.0,            # All features are informative
    outliers=0.1,          # 10% outliers
    rng=123,
)               # For reproducibility

# Fit the model
lr_model = LinearRegressor()
mach = fit!(machine(lr_model, X, y))

# Calculate distances
distances = eos_distances(lr_model, mach.fitresult, X, y)

# Get the RMSE of the model
rmse = sqrt(mean(distances))
println("RMSE: ", rmse)

# Calculate sample weights
alpha = 0.1
weights = eos_weights(distances, alpha)

# Re-fit the model with the new weights
mach_new = fit!(machine(lr_model, X, y, weights))

# Recalculate the RMSE
distances_new = eos_distances(lr_model, mach_new.fitresult, X, y)
rmse_new = sqrt(mean(distances_new))
println("RMSE (new): ", rmse_new)
