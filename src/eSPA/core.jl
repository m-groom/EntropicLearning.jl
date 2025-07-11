# Core functions for fitting an eSPA model

# Initialisation for fitting an eSPA model
function initialise(
    model::eSPAClassifier,
    X::AbstractMatrix{Tf},
    y::AbstractVector{Ti},
    D_features::Int,
    T_instances::Int,
    M_classes::Int,
) where {Tf<:AbstractFloat,Ti<:Integer}
    # Get number of clusters
    K_clusters = model.K
    @assert K_clusters <= T_instances (
        "Number of clusters must be less than or equal to the number of instances"
    )

    # Initialise the random number generator
    rng = EntropicLearning.get_rng(model.random_state)

    # Initialise the feature importance vector
    W = zeros(Tf, D_features)
    if isfinite(model.epsW)
        if model.mi_init
            # Initialise W[d] using the mutural information between feature d and y
            @inbounds for d in 1:D_features
                W[d] = mi_continuous_discrete(view(X, d, :), y; n_neighbors=3, rng=rng)
            end
        else
            rand!(rng, W)
        end
        EntropicLearning.normalise!(W)
    else
        fill!(W, Tf(1.0) / D_features)
    end

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
        iseeds = sample(rng, 1:T_instances, K_clusters; replace=false)
    end
    copyseeds!(C, X, iseeds)

    # Initialise the conditional probability matrix
    L = rand(rng, Tf, M_classes, K_clusters)
    EntropicLearning.left_stochastic!(L)

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
    weights::AbstractVector{Tf},
) where {Tf<:AbstractFloat}
    # Get dimensions
    K_clusters, T_instances = size(G)

    # Compute the discretisation error term
    # disc_error[k, t] = weights[t] * sum_d W[d] × (X[d, t] - C[d, k])^2
    disc_error = Matrix{Tf}(undef, K_clusters, T_instances)
    @inbounds for t in 1:T_instances
        wt = weights[t]
        for k in 1:K_clusters
            temp = Tf(0.0)  # Cache current value for sum
            @simd for d in axes(X, 1)
                temp += W[d] * (X[d, t] - C[d, k])^2
            end
            disc_error[k, t] = wt * temp # Store result back to disc_error
        end
    end

    if epsC > 0
        # Compute the classification error term
        logLP = Matrix{Tf}(undef, K_clusters, T_instances)  # logLP = ε_C × log.(Λ)' × Π
        # For now the weights only modify the discretisation error
        prefactor = Tf(epsC / T_instances)  # TODO: modify this for weighted case
        LinearAlgebra.BLAS.gemm!(
            'T', 'N', prefactor, EntropicLearning.safelog(L; tol=eps(Tf)), P, Tf(0.0), logLP
        )

        # Subtract the classification error term from the discretisation error term
        @inbounds @simd for i in eachindex(disc_error)
            disc_error[i] -= logLP[i]
        end
    end

    # Update Γ
    EntropicLearning.assign_closest!(G, disc_error)  # Updates G.rowval
    return nothing
end

# Find any empty clusters
function find_empty(G::SparseMatrixCSC{Bool,Int})
    sumG = sum(G; dims=2)           # Number of instances in each cluster
    notEmpty = vec(sumG .> eps())   # Find any empty boxes
    K_new = sum(notEmpty)           # Number of non-empty boxes
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
    weights::AbstractVector{Tf},
) where {Tf<:AbstractFloat}
    # Get dimensions
    D_features, T_instances = size(X)

    # Calculate the discretisation error for each feature dimension
    if isfinite(epsW)
        # b[d] will store -sum_t weights[t] * (X[d,t] - C[d,k]×Γ[k, t])^2
        b = zeros(Tf, D_features)

        @inbounds for t in 1:T_instances
            k = G.rowval[t]  # Which cluster instance t belongs to
            wt = weights[t]
            @simd for d in 1:D_features
                b[d] -= wt * (X[d, t] - C[d, k])^2
            end
        end

        # Update W
        EntropicLearning.softmax!(W, b; prefactor=Tf(epsW))
    else
        # Set W to the uniform distribution
        fill!(W, Tf(1.0) / D_features)
    end
    return nothing
end

# Update step for the centroid matrix C
function update_C!(
    C::AbstractMatrix{Tf}, X::AbstractMatrix{Tf}, G::SparseMatrixCSC{Bool,Int}, weights::AbstractVector{Tf}
) where {Tf<:AbstractFloat}
    # Get dimensions
    D_features, T_instances = size(X)
    K_clusters = size(C, 2)

    # Reset the centroids
    fill!(C, Tf(0.0))
    denom = zeros(Tf, K_clusters)

    # Accumulate the numerator and denominator
    @inbounds for t in 1:T_instances
        k = G.rowval[t]             # Cluster assignment for instance t
        wt = weights[t]             # Weight for instance t
        denom[k] += wt              # Accumulate the denominator
        @simd for d in 1:D_features
            C[d, k] += wt * X[d, t] # Accumulate the numerator
        end
    end

    # Apply the denominator
    @inbounds for k in 1:K_clusters
        if denom[k] > 0 # Avoid division by zero
            @simd for d in 1:D_features
                C[d, k] /= denom[k]
            end
        end
    end
    return nothing
end

# Update step for the conditional probability matrix Λ
function update_L!(
    L::AbstractMatrix{Tf}, P::AbstractMatrix{Tf}, G::SparseMatrixCSC{Bool,Int}
) where {Tf<:AbstractFloat}
    # Calculate the new conditional probabilities
    mul!(L, P, G') # Λ = Π × Γ'

    # Normalise
    EntropicLearning.left_stochastic!(L)
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
    weights::AbstractVector{Tf},
) where {Tf<:AbstractFloat}
    # Get dimensions
    D_features, T_instances = size(X)

    # Calculate the discretisation error
    disc_error = Tf(0.0) # sum_tdk weights[t] * W[d] * (X[d, t] - C[d, k]×Γ[k, t])^2
    @inbounds for t in 1:T_instances
        k = G.rowval[t]
        temp = Tf(0.0)
        @simd for d in 1:D_features
            temp += W[d] * (X[d, t] - C[d, k])^2
        end
        disc_error += weights[t] * temp
    end

    # Calculate the classification error - TODO: include contribution from weights
    @inbounds LG = view(L, :, G.rowval)   # LG = Λ × Γ
    class_error = Tf(epsC / T_instances) * EntropicLearning.cross_entropy(P, LG; tol=eps(Tf))

    # Calculate the entropy term
    if isfinite(epsW)
        entr_W = Tf(epsW) * EntropicLearning.entropy(W; tol=eps(Tf)) # Includes the minus sign
    else
        entr_W = Tf(0.0)
    end

    # Calculate the loss
    return disc_error + class_error  - entr_W
end

# Fit function
function _fit!(
    C::AbstractMatrix{Tf},
    W::AbstractVector{Tf},
    L::AbstractMatrix{Tf},
    G::SparseMatrixCSC{Bool,Int},
    model::eSPAClassifier,
    verbosity::Int,
    X::AbstractMatrix{Tf},
    P::AbstractMatrix{Tf},
    weights::AbstractVector{Tf},
    to::TimerOutput,
) where {Tf<:AbstractFloat}

    # --- Initialise Loss ---
    K_current = size(C, 2)                      # Current number of clusters
    loss = fill(Tf(Inf), model.max_iter + 1)    # Loss for each iteration
    iter = 0                                    # Iteration counter
    loss[1] = calc_loss(X, P, C, W, L, G, model.epsC, model.epsW, weights)

    # --- Main Optimisation Loop ---
    @timeit to "Training" begin
        while !converged(loss, iter, model.max_iter, model.tol)
            # Update iteration counter
            iter += 1

            # Evaluation of the Γ-step
            @timeit to "G" update_G!(G, X, P, C, W, L, model.epsC, weights)

            # Discard empty boxes
            notEmpty, K_new = find_empty(G)
            if K_new < K_current
                @timeit to "Prune" C, L, G = remove_empty(C, L, G, notEmpty)
                K_current = copy(K_new)
            end

            # Evaluation of the W-step
            @timeit to "W" update_W!(W, X, C, G, model.epsW, weights)

            # Evaluation of the C-step
            @timeit to "C" update_C!(C, X, G, weights)

            # Evaluation of the Λ-step
            @timeit to "L" update_L!(L, P, G)

            # Update loss
            @timeit to "Loss" loss[iter + 1] = calc_loss(
                X, P, C, W, L, G, model.epsC, model.epsW, weights
            )

            # Check if loss function has increased
            check_loss(loss, iter, verbosity)
        end
    end

    # Warn if the maximum number of iterations was reached
    exceeded = check_iter(iter, model.max_iter, verbosity)

    # --- Unbiasing step ---
    if !exceeded
        @timeit to "Unbias" begin
            # Unbias Γ
            update_G!(G, X, P, C, W, L, Tf(0.0), weights)

            # Discard empty boxes
            notEmpty, K_new = find_empty(G)
            if K_new < K_current
                C, L, G = remove_empty(C, L, G, notEmpty)
                K_current = copy(K_new)
            end

            # Unbias Λ
            update_L!(L, P, G)
        end
    end

    # Return the loss, number of iterations and the timer output
    return loss[2:(iter + 1)], iter, to
end


# Function to check for convergence
function converged(
    loss::AbstractVector{<:AbstractFloat}, iter::Int, max_iter::Int, tol::Float64
)
    # Check if max iterations reached
    if iter >= max_iter
        return true
    end

    # The first iteration (iter=0) is never converged
    if iter == 0
        return false
    end

    # Check for convergence based on relative loss change
    return abs((loss[iter + 1] - loss[iter]) / loss[iter]) <= tol
end

# Function to check if the loss has increased
function check_loss(
    loss::AbstractVector{Tf}, iter::Int, verbosity::Int; context::String=""
) where {Tf<:AbstractFloat}
    if verbosity > 0 && loss[iter + 1] - loss[iter] > eps(Tf)
        msg = isempty(context) ? "" : " in $context"
        @warn "Loss function$msg has increased at iteration $iter by $(loss[iter + 1] - loss[iter])"
    end
    return nothing
end

# Function to check if the maximum number of iterations has been reached
function check_iter(iter::Int, max_iter::Int, verbosity::Int; context::String="")
    exceeded = iter >= max_iter
    if verbosity > 0 && exceeded
        msg = isempty(context) ? "" : " in $context"
        @warn "Maximum number of iterations reached$msg"
    end
    return exceeded
end

# Function to calculate Π
function update_P!(
    P::AbstractMatrix{Tf}, L::AbstractMatrix{Tf}, G::SparseMatrixCSC{Bool,Int}
) where {Tf<:AbstractFloat}
    # Calculate Π = Λ × Γ
    mul!(P, L, G)

    # Ensure Π is normalised
    EntropicLearning.left_stochastic!(P)

    return nothing
end

# Prediction function
function _predict(
    model::eSPAClassifier,
    C::AbstractMatrix{Tf},
    W::AbstractVector{Tf},
    L::AbstractMatrix{Tf},
    X::AbstractMatrix{Tf},
) where {Tf<:AbstractFloat}
    # Get dimensions
    T_instances = size(X, 2)
    K_clusters = size(C, 2)
    M_classes = size(L, 1)

    # Initialise the random number generator
    rng = EntropicLearning.get_rng(model.random_state)

    # Initialise Γ and Π
    G = sparse(
        rand(rng, 1:K_clusters, T_instances),
        1:T_instances,
        ones(Bool, T_instances),
        K_clusters,
        T_instances,
    )
    P = Matrix{Tf}(undef, M_classes, T_instances)
    weights = fill(Tf(1 / T_instances), T_instances)

    # Update Γ
    update_G!(G, X, P, C, W, L, Tf(0.0), weights)

    # Update Π
    update_P!(P, L, G)

    # Return Π
    return P, G
end
