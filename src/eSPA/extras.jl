# This file contains functions that are used in the eSPA module but are not part of the
# core eSPA algorithm.

"""
    compute_mi_cd(c::AbstractVector{Tf}, d::AbstractVector{Ti}, n_neighbors::Int=3) where {Tf<:AbstractFloat,Ti<:Integer}

Compute mutual information between a continuous variable and a discrete variable using the
Ross (2014) estimator.

This function implements the k-nearest neighbor based mutual information estimator
specifically designed for mixed continuous-discrete data. The method uses the Chebyshev (L∞)
distance metric and applies numerical stability improvements to handle edge cases.

# Arguments
- `c::AbstractVector{Tf}`: Samples of a continuous random variable where `Tf<:AbstractFloat`
- `d::AbstractVector{Ti}`: Samples of a discrete random variable where `Ti<:Integer`
- `n_neighbors::Int=3`: Number of nearest neighbors to search for within each discrete
  class.
  Higher values reduce variance but may introduce bias.

# Returns
- `mi::Tf`: Estimated mutual information in nat units. Always non-negative (negatives are
  clipped to 0).

# References
- Ross, B. C. "Mutual Information between Discrete and Continuous Data Sets". PLoS ONE
  9(2), 2014.
"""
function compute_mi_cd(
    c::AbstractVector{Tf}, d::AbstractVector{Ti}, n_neighbors::Int=3
) where {Tf<:AbstractFloat,Ti<:Integer}
    n_samples = length(c)

    # Pre-allocate working arrays
    radius = zeros(Tf, n_samples)
    label_counts = zeros(Ti, n_samples)
    k_all = zeros(Ti, n_samples)

    # Get unique labels
    unique_labels = unique(d)

    # If there's only one unique label, MI is 0
    if length(unique_labels) == 1
        return Tf(0.0)
    end

    # Pre-compute label indices for all labels
    label_to_indices = Dict{Ti,Vector{Int}}()
    @inbounds for i in 1:n_samples
        label = d[i]
        if haskey(label_to_indices, label)
            push!(label_to_indices[label], i)
        else
            label_to_indices[label] = [i]
        end
    end

    # Process each unique label using pre-computed indices
    for label in unique_labels
        label_indices = label_to_indices[label]
        count = length(label_indices)

        if count > 1
            k = min(n_neighbors, count - 1)

            # Extract values for this label
            label_values = Vector{Tf}(undef, count)
            @inbounds @simd for i in 1:count
                label_values[i] = c[label_indices[i]]
            end

            # Create matrix for KDTree (single row since 1D data)
            label_matrix = reshape(label_values, 1, count)
            kdtree = KDTree(label_matrix, Chebyshev())

            # Find k nearest neighbors for each point in this label group
            query_point = Vector{Tf}(undef, 1) # Pre-allocate to avoid repeated allocations
            @inbounds for (i, idx) in enumerate(label_indices)
                query_point[1] = label_values[i]
                _, dists = knn(kdtree, query_point, k + 1)

                # Get maximum distance (k-th neighbor, avoids the need to sort)
                radius[idx] = maximum(dists)
            end

            # Update k_all and label_counts for this label
            @inbounds @simd for idx in label_indices
                k_all[idx] = k
                label_counts[idx] = count
            end
        else
            # Single point labels get count but no valid k
            @inbounds for idx in label_indices
                label_counts[idx] = count
            end
        end
    end

    # Count and filter points
    n_samples_filtered = 0
    @inbounds for i in 1:n_samples
        if label_counts[i] > 1
            n_samples_filtered += 1
        end
    end

    if n_samples_filtered == 0
        return Tf(0.0)
    end

    # Pre-allocate filtered arrays with known size
    label_counts_filtered = Vector{Ti}(undef, n_samples_filtered)
    k_all_filtered = Vector{Ti}(undef, n_samples_filtered)
    c_filtered_values = Vector{Tf}(undef, n_samples_filtered)
    radius_filtered = Vector{Tf}(undef, n_samples_filtered)

    # Fill filtered arrays
    j = 1
    @inbounds for i in 1:n_samples
        if label_counts[i] > 1
            label_counts_filtered[j] = label_counts[i]
            k_all_filtered[j] = k_all[i]
            c_filtered_values[j] = c[i]
            radius_filtered[j] = radius[i]
            j += 1
        end
    end

    # Apply nextafter towards zero to radius
    @inbounds @simd for i in eachindex(radius_filtered)
        if radius_filtered[i] > 0
            radius_filtered[i] = prevfloat(radius_filtered[i])
        end
        # Ensure positive
        radius_filtered[i] = max(radius_filtered[i], eps(Tf))
    end

    # Build KDTree for all filtered points
    c_filtered_matrix = reshape(c_filtered_values, 1, n_samples_filtered)
    kdtree_all = KDTree(c_filtered_matrix, Chebyshev())

    # Count points within radius for each point
    m_all = Vector{Ti}(undef, n_samples_filtered)
    query_point = Vector{Tf}(undef, 1)
    @inbounds for i in 1:n_samples_filtered
        query_point[1] = c_filtered_values[i]
        neighbors = inrange(kdtree_all, query_point, radius_filtered[i])
        m_all[i] = length(neighbors)
    end

    # Compute mutual information using the formula
    mi =
        digamma(n_samples_filtered) + mean(digamma.(k_all_filtered)) -
        mean(digamma.(label_counts_filtered)) - mean(digamma.(m_all))

    return max(Tf(0.0), mi)
end

"""
    mi_continuous_discrete(
        x::AbstractVector{Tf}, y::AbstractVector{Ti}; n_neighbors::Int=3,
        rng::AbstractRNG=Random.default_rng()
    ) where {Tf<:AbstractFloat,Ti<:Integer}

Estimate mutual information between a single continuous feature vector and a discrete target
variable.

This function computes the mutual information (MI) score between a continuous feature and
discrete class labels. The implementation uses the Ross (2014) estimator designed for mixed
continuous-discrete data, with feature scaling and noise addition for numerical stability.

# Arguments
- `x::AbstractVector{Tf}`: Feature vector of length T containing continuous values
- `y::AbstractVector{Ti}`: Target vector of length T containing discrete class labels
- `n_neighbors::Int=3`: Number of neighbors for MI estimation. Higher values reduce
  variance but may introduce bias
- `rng::AbstractRNG=Random.default_rng()`: Random number generator for reproducible
  noise addition

# Returns
- `mi::Tf`: Mutual information score in nat units (always non-negative)

# References
- Ross, B. C. "Mutual Information between Discrete and Continuous Data Sets". PLoS ONE
  9(2), 2014.
"""
function mi_continuous_discrete(
    x::AbstractVector{Tf},
    y::AbstractVector{Ti};
    n_neighbors::Int=3,
    rng::AbstractRNG=Random.default_rng(),
) where {Tf<:AbstractFloat,Ti<:Integer}
    n_samples = length(x)

    # Scale feature (without centering)
    std_val = std(x; corrected=false)
    if std_val > 0
        x_scaled = x ./ std_val  # Use broadcasting instead of copy + in-place division
    else
        x_scaled = copy(x)  # Only copy when necessary
    end

    # Add small noise to continuous feature
    # Following sklearn's approach: noise = 1e-10 * max(1, mean(abs(x))) * randn
    mean_abs = max(Tf(1.0), mean(abs, x_scaled))
    noise_scale = Tf(1e-10) * mean_abs

    # Generate noise and add to feature
    @inbounds @simd for i in 1:n_samples
        x_scaled[i] += noise_scale * randn(rng, Tf)
    end

    # Compute MI
    return compute_mi_cd(x_scaled, y, n_neighbors)
end

"""
    mi_continuous_discrete(
        X::AbstractMatrix{Tf}, y::AbstractVector{Ti}; n_neighbors::Int=3,
        rng::AbstractRNG=Random.default_rng()
    ) where {Tf<:AbstractFloat,Ti<:Integer}

Estimate mutual information between multiple continuous features and a discrete target
variable.

This function computes mutual information (MI) scores for feature selection in machine
learning tasks. MI quantifies the statistical dependency between variables - zero indicates
independence, while higher values indicate stronger dependency. The implementation uses the
Ross (2014) estimator designed for mixed continuous-discrete data.

# Arguments
- `X::AbstractMatrix{Tf}`: Feature matrix of shape (D, T) where D is the number of
  features and T is the number of samples
- `y::AbstractVector{Ti}`: Target vector of length T containing discrete class labels
- `n_neighbors::Int=3`: Number of neighbors for MI estimation. Higher values reduce
  variance but may introduce bias
- `rng::AbstractRNG=Random.default_rng()`: Random number generator for reproducible
  noise addition

# Returns
- `mi::Vector{Tf}`: Mutual information scores for each feature in nat units (always
  non-negative)

# References
- Ross, B. C. "Mutual Information between Discrete and Continuous Data Sets". PLoS ONE
  9(2), 2014.
- Kraskov, A., Stögbauer, H. & Grassberger, P. "Estimating mutual information". Phys. Rev.
  E 69, 066138 (2004).
"""
function mi_continuous_discrete(
    X::AbstractMatrix{Tf},
    y::AbstractVector{Ti};
    n_neighbors::Int=3,
    rng::AbstractRNG=Random.default_rng(),
) where {Tf<:AbstractFloat,Ti<:Integer}
    D = size(X, 1)

    # Pre-allocate result array
    mi_scores = Vector{Tf}(undef, D)

    # Compute MI for each feature
    @inbounds for d in 1:D
        mi_scores[d] = mi_continuous_discrete(
            view(X, d, :), y; n_neighbors=n_neighbors, rng=rng
        )
    end

    return mi_scores
end

# Helper function to get RNG
_get_rng(random_state::Int) = MersenneTwister(random_state)
_get_rng(random_state::AbstractRNG) = random_state
