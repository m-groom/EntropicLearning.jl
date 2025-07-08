using MLJ
using EntropicLearning
using Random
using Statistics
using MLJGLMInterface

# Implementation of eos_distancesfor LinearRegressor from MLJGLMInterface
function EntropicLearning.eos_distances(
    model::MLJGLMInterface.LinearRegressor,
    fitresult,
    X,
    y=nothing
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
        distances = (ŷ .- ŷ_mean).^2
    else
        # For fit: use squared residuals
        y_vec = vec(y)  # Ensure y is a vector
        distances = (y_vec .- ŷ).^2
    end

    return distances
end

# Generate synthetic regression data with outliers
X, y = make_regression(200, 5;               # 200 samples, 5 features
                       noise=0.5,             # Moderate noise
                       sparse=0.0,            # All features are informative
                       outliers=0.1,          # 10% outliers
                       rng=123)               # For reproducibility

# Wrap the model in an EOSWrapper
lr_model = LinearRegressor()
eos_model = EOSWrapper(model=lr_model, alpha=0.1)
mach = machine(eos_model, X, y) |> fit!
