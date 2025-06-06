# This file contains functions that are used in the eSPA module but are not part of the core eSPA algorithm.

"""
    compute_mi_cd(c::AbstractVector{Tf}, d::AbstractVector{Ti}, n_neighbors::Int=3) where {Tf<:AbstractFloat, Ti<:Integer}

Compute mutual information between a continuous variable and a discrete variable using the Ross (2014) estimator.

This function implements the k-nearest neighbor based mutual information estimator specifically designed
for mixed continuous-discrete data. The method uses the Chebyshev (L∞) distance metric and applies
numerical stability improvements to handle edge cases.

# Arguments
- `c::AbstractVector{Tf}`: Samples of a continuous random variable where `Tf<:AbstractFloat`
- `d::AbstractVector{Ti}`: Samples of a discrete random variable where `Ti<:Integer`
- `n_neighbors::Int=3`: Number of nearest neighbors to search for within each discrete class.
  Higher values reduce variance but may introduce bias.

# Returns
- `mi::Tf`: Estimated mutual information in nat units. Always non-negative (negatives are clipped to 0).

# References
- Ross, B. C. "Mutual Information between Discrete and Continuous Data Sets". PLoS ONE 9(2), 2014.
"""
function compute_mi_cd(
    c::AbstractVector{Tf}, d::AbstractVector{Ti}, n_neighbors::Int=3
) where {Tf<:AbstractFloat,Ti<:Integer}
    n_samples = length(c)
    c_reshaped = reshape(c, :, 1)  # Make it a column matrix for KDTree

    radius = zeros(Tf, n_samples)
    label_counts = zeros(Ti, n_samples)
    k_all = zeros(Ti, n_samples)

    # For each unique label
    unique_labels = unique(d)

    # If there's only one unique label, MI is 0
    if length(unique_labels) == 1
        return Tf(0.0)
    end

    for label in unique_labels
        mask = d .== label
        count = sum(mask)

        if count > 1
            k = min(n_neighbors, count - 1)
            # Get indices where mask is true
            masked_indices = findall(mask)
            masked_c = c_reshaped[masked_indices, :]

            # Build KDTree with Chebyshev metric for masked points
            kdtree = KDTree(masked_c', Chebyshev())

            # Find k nearest neighbors for each point in this label group
            for (i, idx) in enumerate(masked_indices)
                idxs, dists = knn(kdtree, vec(masked_c[i, :]), k + 1)  # k+1 because it includes the point itself

                # Sort distances to ensure correct ordering
                sorted_dists = sort(dists)
                # The k-th neighbor (excluding self which has distance 0) is at position k+1
                radius[idx] = sorted_dists[end]
            end

            k_all[mask] .= k
        end
        label_counts[mask] .= count
    end

    # Ignore points with unique labels
    mask = label_counts .> 1
    n_samples_filtered = sum(mask)

    if n_samples_filtered == 0
        return Tf(0.0)
    end

    label_counts_filtered = label_counts[mask]
    k_all_filtered = k_all[mask]
    c_filtered = c_reshaped[mask, :]
    radius_filtered = radius[mask]

    # Apply nextafter towards zero to radius
    for i in eachindex(radius_filtered)
        # Mimic np.nextafter(radius, 0) - move slightly towards zero
        if radius_filtered[i] > 0
            radius_filtered[i] = prevfloat(radius_filtered[i])
        end
        # Ensure positive
        radius_filtered[i] = max(radius_filtered[i], eps(Tf))
    end

    # Build KDTree for all filtered points
    kdtree_all = KDTree(c_filtered', Chebyshev())

    # Count points within radius for each point
    m_all = zeros(Ti, n_samples_filtered)
    for i in 1:n_samples_filtered
        # Use inrange to find all points within radius
        neighbors = inrange(kdtree_all, vec(c_filtered[i, :]), radius_filtered[i])
        m_all[i] = length(neighbors)  # Count includes the point itself
    end

    # Compute mutual information using the formula
    mi =
        digamma(n_samples_filtered) + mean(digamma.(k_all_filtered)) -
        mean(digamma.(label_counts_filtered)) - mean(digamma.(m_all))

    return max(Tf(0.0), mi)
end

"""
    mi_continuous_discrete(X::AbstractMatrix{Tf}, y::AbstractVector{Ti}; n_neighbors::Int=3, rng::AbstractRNG=Random.default_rng()) where {Tf<:AbstractFloat, Ti<:Integer}

Estimate mutual information between multiple continuous features and a discrete target variable.

This function computes mutual information (MI) scores for feature selection in machine learning tasks.
MI quantifies the statistical dependency between variables - zero indicates independence, while higher
values indicate stronger dependency. The implementation uses the Ross (2014) estimator designed for
mixed continuous-discrete data.

# Arguments
- `X::AbstractMatrix{Tf}`: Feature matrix of shape (D, T) where D is the number of features and T is the number of samples
- `y::AbstractVector{Ti}`: Target vector of length T containing discrete class labels
- `n_neighbors::Int=3`: Number of neighbors for MI estimation. Higher values reduce variance but may introduce bias
- `rng::AbstractRNG=Random.default_rng()`: Random number generator for reproducible noise addition

# Returns
- `mi::Vector{Tf}`: Mutual information scores for each feature in nat units (always non-negative)

# References
- Ross, B. C. "Mutual Information between Discrete and Continuous Data Sets". PLoS ONE 9(2), 2014.
- Kraskov, A., Stögbauer, H. & Grassberger, P. "Estimating mutual information". Phys. Rev. E 69, 066138 (2004).
"""
function mi_continuous_discrete(
    X::AbstractMatrix{Tf},
    y::AbstractVector{Ti};
    n_neighbors::Int=3,
    rng::AbstractRNG=Random.default_rng(),
) where {Tf<:AbstractFloat,Ti<:Integer}
    # Get dimensions
    D, T = size(X)

    # Scale features (without centering, similar to sklearn's scale with with_mean=False)
    X_scaled = copy(X)
    @views for d in 1:D
        std_val = std(X[d, :], corrected=false)
        if std_val > 0
            X_scaled[d, :] ./= std_val
        end
    end

    # Add small noise to continuous features
    # Following sklearn's approach: noise = 1e-10 * max(1, mean(abs(x))) * randn
    for d in 1:D
        mean_abs = max(Tf(1.0), mean(abs.(X_scaled[d, :])))
        X_scaled[d, :] .+= 1e-10 * mean_abs * randn(rng, T)
    end

    # Compute MI for each feature
    mi_scores = zeros(Tf, D)
    @views for d in 1:D
        mi_scores[d] = compute_mi_cd(X_scaled[d, :], y, n_neighbors)
    end

    return mi_scores
end
