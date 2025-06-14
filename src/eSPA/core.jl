# Core functions for fitting an eSPA model

# Initialisation for fitting an eSPA model
function initialise(
    model::eSPAClassifier,
    X::AbstractMatrix{Tf},
    y::AbstractVector{Ti},
    D_features::Int,
    T_instances::Int,
    M_classes::Int;
    rng::AbstractRNG=Random.default_rng(),
) where {Tf<:AbstractFloat,Ti<:Integer}
    # Get number of clusters
    K_clusters = model.K
    @assert K_clusters <= T_instances (
        "Number of clusters must be less than or equal to the number of instances"
    )

    # Initialise the feature importance vector
    W = zeros(Tf, D_features)
    if model.mi_init
        # Initialise W[d] using the mutural information between feature d and y
        @inbounds for d in 1:D_features
            W[d] = mi_continuous_discrete(view(X, d, :), y; n_neighbors=3, rng=rng)
        end
    else
        rand!(rng, W)
    end
    normalise!(W)

    # Initialise the centroid matrix
    C = zeros(Tf, D_features, K_clusters)
    if model.kpp_init   # Use k-means++ to initialise the centroids
        iseeds = Vector{Int}(undef, K_clusters)
        if model.mi_init    # We already have a somewhat informative W
            initseeds!(iseeds, KmppAlg(), X, WeightedSqEuclidean(W); rng=rng)
        else
            initseeds!(iseeds, KmppAlg(), X, SqEuclidean(); rng=rng)
        end
    else    # Randomly select K data points as centroids
        iseeds = StatsBase.sample(rng, 1:T_instances, K_clusters; replace=false)
    end
    copyseeds!(C, X, iseeds)

    # Initialise the conditional probability matrix
    L = rand(rng, Tf, M_classes, K_clusters)
    left_stochastic!(L)

    # Initialise the affiliation matrix
    G = sparse(
        rand(rng, 1:K_clusters, T_instances),
        1:T_instances,
        ones(Bool, T_instances),
        K_clusters,
        T_instances,
    )

    return C, W, L, G
end

# Update step for the affiliation matrix Γ
function update_G!(
    G::SparseMatrixCSC{Bool,Int},
    X::AbstractMatrix{Tf},
    P::AbstractMatrix{Tf},
    C::AbstractMatrix{Tf},
    W::AbstractVector{Tf},
    L::AbstractMatrix{Tf},
    epsC::Float64,
) where {Tf<:AbstractFloat}
    # Get dimensions
    K_clusters, T_instances = size(G)

    # Compute the discretisation error term: disc_error[k, t] = sum_d W[d] × (X[d, t] - C[d, k])^2
    disc_error = Matrix{Tf}(undef, K_clusters, T_instances)
    @inbounds for t in 1:T_instances
        for k in 1:K_clusters
            temp = Tf(0.0)  # Cache current value for sum
            @simd for d in axes(X, 1)
                temp += W[d] * (X[d, t] - C[d, k])^2
            end
            disc_error[k, t] = temp # Store result back to disc_error
        end
    end

    if epsC > 0
        # Compute the classification error term
        logLP = Matrix{Tf}(undef, K_clusters, T_instances)  # logLP = ε_C × log.(Λ)' × Π
        LinearAlgebra.BLAS.gemm!(
            'T', 'N', Tf(epsC), safelog(L; tol=eps(Tf)), P, Tf(0.0), logLP
        )

        # Subtract the classification error term from the discretisation error term
        @inbounds @simd for i in eachindex(disc_error)
            disc_error[i] -= logLP[i]
        end
    end

    # Update Γ
    assign_closest!(G, disc_error)  # Updates G.rowval
    return nothing
end

# Find any empty clusters
function find_empty(G::SparseMatrixCSC{Bool,Int})
    sumG = sum(G; dims=2)       # Number of instances in each cluster
    notEmpty = sumG .> eps()  # Find any empty boxes
    K_new = sum(notEmpty)       # Number of non-empty boxes
    return notEmpty, K_new
end

# Remove empty clusters from C, Λ and Γ
function remove_empty(
    C_::AbstractMatrix{Tf},
    L_::AbstractMatrix{Tf},
    G_::SparseMatrixCSC{Bool,Int},
    idx::BitVector,
) where {Tf<:AbstractFloat}
    @inbounds begin
        C = C_[:, idx]
        L = L_[:, idx]
        G = G_[idx, :]
    end
    return C, L, G
end

# Update step for the feature importance vector W
function update_W!(
    W::AbstractVector{Tf},
    X::AbstractMatrix{Tf},
    C::AbstractMatrix{Tf},
    G::SparseMatrixCSC{Bool,Int},
    epsW::Float64,
) where {Tf<:AbstractFloat}
    # Get dimensions
    D_features, T_instances = size(X)

    # Calculate the discretisation error for each feature dimension
    b = zeros(Tf, D_features)   # b[d] will store -sum_t sum_k (X[d,t] - C[d,k]×Γ[k, t])^2
    # Iterate over instances (columns of X)
    @inbounds for t in 1:T_instances
        cluster_idx = G.rowval[t]  # Which cluster instance t belongs to
        @simd for d in 1:D_features
            b[d] -= (X[d, t] - C[d, cluster_idx])^2
        end
    end

    # Update W
    softmax!(W, b; prefactor=Tf(T_instances * epsW))
    return nothing
end

# Update step for the centroid matrix C
function update_C!(
    C::AbstractMatrix{Tf}, X::AbstractMatrix{Tf}, G::SparseMatrixCSC{Bool,Int}
) where {Tf<:AbstractFloat}
    # Calculate the new centroids
    mul!(C, X, G')  # C = X × Γ'

    # Average over the number of instances in each cluster
    # Clusters are guaranteed to be non-empty if remove_empty has been called first
    C ./= sum(G; dims=2)'
    return nothing
end

# Update step for the conditional probability matrix Λ
function update_L!(
    L::AbstractMatrix{Tf}, P::AbstractMatrix{Tf}, G::SparseMatrixCSC{Bool,Int}
) where {Tf<:AbstractFloat}
    # Calculate the new conditional probabilities
    mul!(L, P, G') # Λ = Π × Γ'

    # Normalise
    left_stochastic!(L)
    return nothing
end

# Loss function calculation
function calc_loss(
    X::AbstractMatrix{Tf},
    P::AbstractMatrix{Tf},
    C::AbstractMatrix{Tf},
    W::AbstractVector{Tf},
    L::AbstractMatrix{Tf},
    G::SparseMatrixCSC{Bool,Int},
    epsC::Float64,
    epsW::Float64,
) where {Tf<:AbstractFloat}
    # Get dimensions
    D_features, T_instances = size(X)

    # Calculate the discretisation error
    disc_error = Tf(0.0) # = sum_t sum_d sum_k W[d] * (X[d, t] - C[d, k]×Γ[k, t])^2
    @inbounds for t in 1:T_instances
        cluster_idx = G.rowval[t]
        @simd for d in 1:D_features
            disc_error += W[d] * (X[d, t] - C[d, cluster_idx])^2
        end
    end

    # Calculate the classification error
    @inbounds LG = view(L, :, G.rowval)   # LG = Λ × Γ
    class_error = Tf(epsC) * cross_entropy(P, LG; tol=eps(Tf)) # Includes the minus sign

    # Calculate the entropy term
    entr_W = Tf(epsW) * entropy(W; tol=eps(Tf))                # Includes the minus sign

    # Calculate the loss
    return (disc_error + class_error) / T_instances - entr_W
end

# Function to calculate Π
function update_P!(
    P::AbstractMatrix{Tf}, L::AbstractMatrix{Tf}, G::SparseMatrixCSC{Bool,Int}
) where {Tf<:AbstractFloat}
    # From entlearn:
    # P = assign_closest(-safelog(L; tol=eps(Tf)) * G)
    # Calculate Π = Λ × Γ
    mul!(P, L, G)

    # Ensure Π is normalised
    left_stochastic!(P)

    return nothing
end

# Prediction function
function _predict_proba(
    model::eSPAClassifier, fitresult::eSPAFitResult, X::AbstractMatrix{Tf}
) where {Tf<:AbstractFloat}
    # Get dimensions
    T_instances = size(X, 2)
    K_clusters = size(fitresult.C, 2)
    M_classes = size(fitresult.L, 1)

    # Initialise the random number generator
    rng = _get_rng(model.random_state)

    # Initialise Γ and Π
    G = sparse(
        rand(rng, 1:K_clusters, T_instances),
        1:T_instances,
        ones(Bool, T_instances),
        K_clusters,
        T_instances,
    )
    P = Matrix{Tf}(undef, M_classes, T_instances)

    # Update Γ
    update_G!(G, X, P, fitresult.C, fitresult.W, fitresult.L, Tf(0.0))

    # Update Π
    update_P!(P, fitresult.L, G)

    if model.iterative_pred
        # TODO: implement iterative prediction
        error("Iterative prediction not yet implemented")
    end

    # Return Π (TODO: also return Γ and store it in the report)
    return P
end
