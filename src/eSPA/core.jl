# Core functions for fitting an eSPA model

# Initialisation for fitting an eSPA model
function initialise(model::eSPA, X::AbstractMatrix{Tf}, P::AbstractMatrix{Tf}, y::AbstractVector{Ti}, D_features::Int, T_instances::Int, M_classes::Int; rng::AbstractRNG=Random.default_rng()) where {Tf<:AbstractFloat, Ti<:Integer}
    # Get number of clusters
    K_clusters = model.K
    @assert K_clusters <= T_instances "Number of clusters must be less than or equal to the number of instances"

    # Initialise the centroid matrix
    C = zeros(Tf, D_features, K_clusters)
    if model.kpp_init   # Use k-means++ to initialise the centroids
        iseeds = Vector{Int}(undef, K_clusters)
        initseeds!(iseeds, KmppAlg(), X, SqEuclidean(), rng=rng)
        # initseeds!(iseeds, KmppAlg(), X, WeightedSqEuclidean(W), rng=rng)
        
    else    # Randomly select K data points as centroids
        iseeds = StatsBase.sample(rng, 1:T_instances, K_clusters, replace=false)
    end
    copyseeds!(C, X, iseeds)

    # Initialise the feature importance vector
    W_= zeros(Tf, D_features)

    if model.mi_init
        # Initialise W[d] using the mutural information between feature d and y
    else
        rand!(rng, W_)
        sum_W = sum(W_)
        if sum_W > eps(Tf)
            W_ ./= sum_W
        else
            fill!(W_, 1.0 / D_features)
        end
    end


    G_ = if K_current > 0 && T_instances > 0
        temp_G = sparse(ones(Int, T_instances), 1:T_instances, true, K_current, T_instances)
        _init_G_distances = pairwise(W_metric, C_, X_mat_transposed)
        assign_closest!(temp_G, _init_G_distances)
        temp_G
    else
        spzeros(Bool, K_current, T_instances)
    end

    L_ = zeros(Float64, M_classes, K_current)
    if K_current > 0 && T_instances > 0
        update_L!(L_, Pi_mat, G_, K_current, M_classes)
    end

    initial_loss = if model.max_iter > 0 || model.debug_loss # Calculate if loop will run or debug is on
        calc_loss(X_mat_transposed, Pi_mat, C_, W_, L_, G_, model.epsC, model.epsW, K_current, T_instances)
    else
        Inf
    end

    return C_, W_, G_, L_, W_metric, K_current, initial_loss
end

# Update step for the affiliation matrix Γ
function update_G!(G::SparseMatrixCSC{Bool,Int}, X::AbstractMatrix{Tf}, P::AbstractMatrix{Tf}, C::AbstractMatrix{Tf}, W::AbstractVector{Tf}, L::AbstractMatrix{Tf}, epsC::Float64) where {Tf<:AbstractFloat}
    # Get dimensions
    K_clusters, T_instances = size(G)

    # Compute the discretisation error term
    disc_error = fill(Tf(0.0), K_clusters, T_instances)  # disc_error[k, t] = sum_d W[d] × (X[d, t] - C[d, k])^2
    @inbounds for t in 1:T_instances
        for k in 1:K_clusters
            temp = Tf(0.0)  # Cache current value for sum
            @simd for d in axes(X, 1)
                temp += W[d] * (X[d, t] - C[d, k])^2
            end
            disc_error[k, t] = temp # Store result back to disc_error
        end
    end

    # Compute the classification error term
    logLP = fill(Tf(0.0), K_clusters, T_instances)  # logLP = ε_C × log.(Λ)' × Π
    LinearAlgebra.BLAS.gemm!('T', 'N', Tf(epsC), safelog(L, tol=eps(Tf)), P, Tf(0.0), logLP) # Ensure type stability

    # Subtract the classification error term from the discretisation error term
    @inbounds @simd for i in eachindex(disc_error)
        disc_error[i] -= logLP[i]
    end

    # Update Γ
    assign_closest!(G, disc_error)  # Updates G.rowval
    return nothing
end

# Remove empty clusters from C, Λ and Γ - TODO: improve this
function remove_empty(C_::AbstractMatrix{Tf}, L_::AbstractMatrix{Tf}, G_::SparseMatrixCSC{Bool,Int}, K_current_ref::Ref{Int}) where {Tf<:AbstractFloat}
    if K_current_ref[] <= 0 # No clusters to remove
        return C_, L_, G_
    end

    cluster_sums = sum(G_, dims=2)
    empty_clusters_mask = cluster_sums .< eps(Tf) # Boolean vector of length K

    @inbounds if any(empty_clusters_mask)
        non_empty_mask = .!empty_clusters_mask
        C = C_[:, non_empty_mask]
        L = L_[:, non_empty_mask]
        G = G_[non_empty_mask, :]
        K_current_ref[] = size(C, 2)
    end
    return C, L, G
end

# Update step for the feature importance vector W
function update_W!(W::AbstractVector{Tf}, X::AbstractMatrix{Tf}, C::AbstractMatrix{Tf}, G::SparseMatrixCSC{Bool,Int}, epsW::Float64) where {Tf<:AbstractFloat}
    # Get dimensions
    D_features, T_instances = size(X)

    # Calculate the discretisation error for each feature dimension
    b = zeros(Tf, D_features)   # b[d] will store -sum_t sum_k (X[d,t] - C[d,k]×Γ[k, t])^2
    CG = view(C, :, G.rowval)  # CG = C × Γ

    # Iterate over instances (columns of X)
    @inbounds for t in 1:T_instances
        @simd for d in 1:D_features
            b[d] -= (X[d, t] - CG[d, t])^2
        end
    end

    # Update W
    softmax!(W, b, prefactor=Tf(T_instances * epsW))
    return nothing
end

# Update step for the centroid matrix C
function update_C!(C::AbstractMatrix{Tf}, X::AbstractMatrix{Tf}, G::SparseMatrixCSC{Bool,Int}) where {Tf<:AbstractFloat}
    # Calculate the new centroids
    mul!(C, X, G')  # C = X × Γ'

    # Average over the number of instances in each cluster 
    C ./= sum(G, dims=2)'   # Clusters are guaranteed to be non-empty if remove_empty has been called first
    return nothing
end

# Update step for the conditional probability matrix Λ
function update_L!(L::AbstractMatrix{Tf}, P::AbstractMatrix{Tf}, G::SparseMatrixCSC{Bool,Int}) where {Tf<:AbstractFloat}
    # Calculate the new conditional probabilities
    mul!(L, P, G') # Λ = Π × Γ'

    # Normalise
    left_stochastic!(L)
    return nothing
end

# Loss function calculation
function calc_loss(X::AbstractMatrix{Tf}, P::AbstractMatrix{Tf}, C::AbstractMatrix{Tf}, W::AbstractVector{Tf}, L::AbstractMatrix{Tf}, G::SparseMatrixCSC{Bool,Int}, epsC::Float64, epsW::Float64) where {Tf<:AbstractFloat}
    # Get dimensions
    D_features, T_instances = size(X)

    # Calculate the discretisation error
    disc_error = Tf(0.0)        # disc_error = sum_t sum_d sum_k W[d] * (X[d, t] - C[d, k]×Γ[k, t])^2
    CG = view(C, :, G.rowval)  # CG = C × Γ
    @inbounds for t in 1:T_instances
        @simd for d in 1:D_features
            disc_error += W[d] * (X[d, t] - CG[d, t])^2
        end
    end

    # Calculate the classification error
    LG = view(L, :, G.rowval)   # LG = Λ × Γ
    class_error = Tf(epsC) * cross_entropy(P, LG, tol=eps(Tf))    # Already includes the minus sign

    # Calculate the entropy term
    entr_W = Tf(epsW) * entropy(W, tol=eps(Tf))     # Already includes the minus sign

    # Calculate the loss
    return (disc_error + class_error) / T_instances - entr_W
end

# TODO: add _predict_proba function