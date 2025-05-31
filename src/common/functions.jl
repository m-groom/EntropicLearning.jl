# General functions used across eSPA, SPARTAN and EON

const smallest = eps()
const smaller = 1e4 * smallest
const small = 1e4 * smaller

"""
    safelog(x; tol=smallest)

Computes the natural logarithm of `x` (or each element of `x` if `x` is an `AbstractVector`)
safely by ensuring the argument to `log` is at least `tol`.

If `x` (or an element of `x`) is less than `tol`, `log(tol)` is computed instead.
This helps to avoid errors when `x` (or its elements) are zero or negative.

# Arguments
- `x::Union{Real, AbstractVector{<:Real}}`: The input value or vector of values.

# Keyword Arguments
- `tol::Real`: The tolerance level. Values of `x` (or its elements) below `tol` will be
  replaced by `tol` before taking the logarithm. Defaults to `smallest` (which is `eps()`
  in this context, as defined in `src/common/functions.jl`).

# Returns
- `Real` or `AbstractVector{<:Real}`: The natural logarithm of `max(x, tol)` (or `max.(x, tol)` for vectors).
  The return type generally matches the input type (scalar or vector, preserving vector type if possible).

"""
safelog(x::Tr; tol=smallest) where {Tr<:Real} = log(max(x, tol))
safelog(x::AbstractVector{Tr}; tol=smallest) where {Tr<:Real} = log.(max.(x, tol))


"""
    entropy(W::T; tol=smallest) where {T<:AbstractArray}

Computes the Shannon entropy H(W) = -∑ᵢ Wᵢ log(Wᵢ) for an input array `W`.

The computation uses `safelog` to handle cases where elements of `W` might be zero
or very small, ensuring numerical stability.

# Arguments
- `W::T`: An `AbstractArray` (e.g., vector or matrix) of probabilities or
  weights. The elements of `W` should ideally sum to 1 if representing a
  probability distribution, but the function will compute the entropy regardless.
  `T` can be any subtype of `AbstractArray`.

# Keyword Arguments
- `tol::Real`: Tolerance level passed to `safelog`. Values of `W[i]` below `tol`
  will be replaced by `tol` before taking the logarithm. Defaults to `smallest`.

# Returns
- `Float64`: The computed Shannon entropy of `W`.

"""
function entropy(W::T; tol=smallest) where {T<:AbstractArray}
  H = 0.0
  @inbounds for d in eachindex(W)
    H += W[d] * safelog(W[d]; tol=tol)
  end
  return -H
end


"""
    assign_closest(distances::AbstractMatrix{Tr}) where {Tr<:Real}
    assign_closest(distances::AbstractVector{Tr}) where {Tr<:Real}

Finds the index of the minimum value.
- For a `distances` matrix, it finds the row index of the minimum value in each column.
- For a `distances` vector, it finds the index of the minimum value in the vector.

This function is typically used to assign items to clusters.
When `distances` is a matrix, each column corresponds to an item, and each row
corresponds to a cluster. The value `distances[j, i]` represents the distance
from item `i` to cluster `j`. The function returns a vector where the k-th
element is the index of the cluster closest to item k.

When `distances` is a vector, it can be seen as distances to a single item from
multiple reference points, and the function returns the index of the closest
reference point.

# Arguments
- `distances::AbstractMatrix{Tr}`: A matrix of distances. Each column represents an
  item, and each row represents a reference point (e.g., a cluster centroid).
- `distances::AbstractVector{Tr}`: A vector of distances from multiple reference
  points to a single item.

# Returns
- `Vector{<:Integer}` (for matrix input): A vector where the i-th element is the row
  index (1-based) of the minimum value in the i-th column of `distances`. This
  represents the assignment of each item to its closest reference point.
- `Integer` (for vector input): The index (1-based) of the minimum value in the
  `distances` vector.
"""
function assign_closest(distances::AbstractMatrix{Tr}) where {Tr<:Real}
  return argmin.(eachcol(distances))
end
function assign_closest(distances::AbstractVector{Tr}) where {Tr<:Real}
  return argmin(distances)
end

"""
    assign_closest!(Gamma::AbstractMatrix{T}, distances::AbstractMatrix{Tr}) where {T<:Real, Tr<:Real}

Modifies the `Gamma` matrix to represent the assignment of items to their closest
reference points based on the `distances` matrix. This version is for dense `Gamma`
matrices.

For each item (column in `distances`), the function finds the closest reference point
(row in `distances`) and sets the corresponding entry in `Gamma` to `one(T)`, while
all other entries in that column of `Gamma` are set to `zero(T)`.

`Gamma` is modified in-place.

# Arguments
- `Gamma::AbstractMatrix{T}`: The assignment matrix to be modified. It should have
  dimensions `(number_of_reference_points, number_of_items)`.
  After the function call, `Gamma[j, i]` will be `one(T)` if item `i` is assigned to
  reference point `j`, and `zero(T)` otherwise.
- `distances::AbstractMatrix{Tr}`: A matrix of distances. Each column represents an
  item, and each row represents a reference point.

# Returns
- `nothing`: The function modifies `Gamma` in-place.

# See Also
- [`assign_closest`](@ref): The internal function that computes the assignments.
- [`assign_closest!(::SparseMatrixCSC, ::AbstractMatrix)`](@ref): The method for
  sparse assignment matrices.
"""
function assign_closest!(Gamma::AbstractMatrix{T}, distances::AbstractMatrix{Tr}) where {T<:Real,Tr<:Real}
  assignments = assign_closest(distances)
  fill!(Gamma, zero(T))
  @inbounds for (i, j) in enumerate(assignments)
    Gamma[j, i] = one(T)
  end
  return nothing
end

"""
    assign_closest!(Gamma::SparseMatrixCSC{Tb, Ti}, distances::AbstractMatrix{Tr}) where {Tb<:Bool, Ti<:Integer, Tr<:Real}

Modifies the sparse assignment matrix `Gamma` in-place to reflect the closest
assignments based on the `distances` matrix.

This function assumes `Gamma` is a sparse matrix where `Gamma[j, i] = true` (or 1)
indicates that item `i` is assigned to cluster `j`, and `false` (or 0) otherwise.
It directly modifies the `rowval` field of the sparse matrix `Gamma`.
Specifically, for each column `i` (representing an item), it finds the row index `j`
(representing a cluster) that minimizes `distances[j, i]`. It then sets the `i`-th
element of `Gamma.rowval` to this `j`.

This method is efficient for sparse matrices as it only updates the `rowval` array,
which stores the row indices of the non-zero elements.

**Important:** This function assumes that `Gamma` is structured such that each column
 has exactly one non-zero entry (or one entry that will become non-zero after the
 assignment). The `colptr` field of `Gamma` should reflect this structure.

# Arguments
- `Gamma::SparseMatrixCSC{Tb, Ti}`: The sparse assignment matrix to be modified.
  `Tb` is typically `Bool` and `Ti` is an `Integer` type.
  The `rowval` field of `Gamma` is updated in-place.
- `distances::AbstractMatrix{Tr}`: A matrix of distances, where `Tr` is a `Real` type.
  `distances[j, i]` is the distance from item `i` to cluster `j`.

# Returns
- `nothing`: The function modifies `Gamma` in-place.

# See Also
- [`assign_closest`](@ref): The internal function that computes the assignments.
- [`assign_closest!(::AbstractMatrix, ::AbstractMatrix)`](@ref): The method for
  dense assignment matrices.
"""
function assign_closest!(Gamma::SparseMatrixCSC{Tb,Ti}, distances::AbstractMatrix{Tr}) where {Tb<:Bool,Ti<:Integer,Tr<:Real}
  Gamma.rowval .= assign_closest(distances)
  return nothing
end


"""
    left_stochastic(A::AbstractMatrix{Tr}) where {Tr<:Real}

Normalizes the columns of matrix `A` so that each column sums to 1.

This creates a new matrix where each element `A[i,j]` is divided by the sum
of the j-th column. The original matrix `A` is not modified.

# Arguments
- `A::AbstractMatrix{Tr}`: The input matrix. `Tr` is a `Real` type.

# Returns
- `AbstractMatrix{<:Real}`: A new matrix with columns normalized to sum to 1.
  The element type will be the result of the division (often `Float64`).

# See Also
- [`left_stochastic!`](@ref): In-place version of this function.
- [`right_stochastic`](@ref): Normalizes rows to sum to 1.
- [`right_stochastic!`](@ref): In-place version of `right_stochastic`.
"""
left_stochastic(A::AbstractMatrix{Tr}) where {Tr<:Real} = A ./ sum(A, dims=1)

"""
    left_stochastic!(A::AbstractMatrix{Tr}) where {Tr<:Real}

Normalizes the columns of matrix `A` in-place so that each column sums to 1.

Each element `A[i,j]` is divided by the sum of the j-th column. The matrix `A`
is modified directly.

# Arguments
- `A::AbstractMatrix{Tr}`: The matrix to be normalized in-place. `Tr` is a `Real` type.

# Returns
- `AbstractMatrix{<:Real}`: The modified matrix `A` with columns normalized.

# See Also
- [`left_stochastic`](@ref): Non-mutating version of this function.
- [`right_stochastic`](@ref): Normalizes rows to sum to 1.
- [`right_stochastic!`](@ref): In-place version of `right_stochastic`.
"""
left_stochastic!(A::AbstractMatrix{Tr}) where {Tr<:Real} = A ./= sum(A, dims=1)

"""
    right_stochastic(A::AbstractMatrix{Tr}) where {Tr<:Real}

Normalizes the rows of matrix `A` so that each row sums to 1.

This creates a new matrix where each element `A[i,j]` is divided by the sum
of the i-th row. The original matrix `A` is not modified.

# Arguments
- `A::AbstractMatrix{Tr}`: The input matrix. `Tr` is a `Real` type.

# Returns
- `AbstractMatrix{<:Real}`: A new matrix with rows normalized to sum to 1.
  The element type will be the result of the division (often `Float64`).

# See Also
- [`right_stochastic!`](@ref): In-place version of this function.
- [`left_stochastic`](@ref): Normalizes columns to sum to 1.
- [`left_stochastic!`](@ref): In-place version of `left_stochastic`.
"""
right_stochastic(A::AbstractMatrix{Tr}) where {Tr<:Real} = A ./ sum(A, dims=2)

"""
    right_stochastic!(A::AbstractMatrix{Tr}) where {Tr<:Real}

Normalizes the rows of matrix `A` in-place so that each row sums to 1.

Each element `A[i,j]` is divided by the sum of the i-th row. The matrix `A`
is modified directly.

# Arguments
- `A::AbstractMatrix{Tr}`: The matrix to be normalized in-place. `Tr` is a `Real` type.

# Returns
- `AbstractMatrix{<:Real}`: The modified matrix `A` with rows normalized.

# See Also
- [`right_stochastic`](@ref): Non-mutating version of this function.
- [`left_stochastic`](@ref): Normalizes columns to sum to 1.
- [`left_stochastic!`](@ref): In-place version of `left_stochastic`.
"""
right_stochastic!(A::AbstractMatrix{Tr}) where {Tr<:Real} = A ./= sum(A, dims=2)


"""
    softmax!(G, A; prefactor=1.0)
    softmax!(A; prefactor=1.0)
    softmax!(W, b; prefactor=1.0)
    softmax!(b; prefactor=1.0)

Computes the softmax function in-place.

For matrix inputs, the softmax is computed column-wise. The result is stored in `G` (if provided),
and the input `A` (or `b` for vectors) is scaled by `prefactor` *in-place*.

For vector inputs, the softmax is computed over the vector elements. The result is stored in `W`
(if provided), and the input `b` is scaled by `prefactor` *in-place*.

If the output argument (`G` or `W`) is not provided, a new array is allocated for it, and the
input array (`A` or `b`) is still modified in-place before the softmax computation. The newly
allocated and computed array is then returned.

This function includes robust handling for empty inputs, columns/vectors with all `-Inf` values
(resulting in a uniform distribution), and cases where the sum of exponentials is zero, `NaN`,
or `Inf` (also resulting in a uniform distribution).

# Arguments
- `G::AbstractMatrix{Tf}`: (Optional) The matrix to store the result for matrix inputs.
  Its dimensions must match `A`.
- `A::AbstractMatrix{Tf}`: The input matrix. **This matrix is modified in-place.**
- `W::AbstractVector{Tf}`: (Optional) The vector to store the result for vector inputs.
  Its length must match `b`.
- `b::AbstractVector{Tf}`: The input vector. **This vector is modified in-place.**

# Keyword Arguments
- `prefactor::Tf`: A positive scaling factor applied to the input array elements
  before the `exp` operation. Defaults to `Tf(1.0)`. An `ArgumentError` is thrown if `prefactor`
  is not positive.

# Returns
- `nothing` if `G` or `W` is provided (results are stored in-place).
- A new `AbstractMatrix{Tf}` or `AbstractVector{Tf}` containing the softmax result if `G` or `W`
  is not provided.

# See Also
- [`softmax`](@ref): Non-mutating version of this function.
"""
function softmax!(G::AbstractMatrix{Tf}, A::AbstractMatrix{Tf}; prefactor::Tf=Tf(1.0)) where {Tf<:AbstractFloat}
  @assert size(G) == size(A) "Size of G and A must be the same"
  if prefactor <= zero(Tf)
    throw(ArgumentError("prefactor must be positive"))
  end

  if size(A, 1) == 0 # No elements to compute softmax over in each column
    return nothing
  end

  # A is modified in place by this operation.
  A ./= prefactor

  @inbounds for t in axes(A, 2)
    # Get a view of the current column of A (which is already scaled)
    current_col_A = view(A, :, t)
    # Get a view of the current column of G for modification
    current_col_G = view(G, :, t)

    max_col_A = maximum(current_col_A)

    if max_col_A == Tf(-Inf)
      # All elements in the original column were -Inf or became -Inf after scaling.
      # Softmax is taken as uniform in this ambiguous case.
      current_col_G .= Tf(1.0) / size(A, 1)
      continue # Move to the next column
    end

    # Compute the exponentials
    @simd for k in axes(A, 1)
      current_col_G[k] = exp(A[k, t] - max_col_A)
    end

    # Normalise
    sum_col_G = sum(current_col_G)

    # Handle cases where sum_col_G is zero (all underflowed), negative (should not happen), NaN, or Inf.
    if sum_col_G <= eps(Tf) || !isfinite(sum_col_G)
      # Fallback to a uniform distribution for this column.
      current_col_G .= Tf(1.0) / size(A, 1)
    else
      # Standard normalisation
      @simd for k in axes(G, 1)
        current_col_G[k] /= sum_col_G
      end
    end
  end
  return nothing
end

# Mutating softmax - matrix input (G not provided)
function softmax!(A::AbstractMatrix{Tf}; prefactor::Tf=Tf(1.0)) where {Tf<:AbstractFloat}
  G = similar(A)
  softmax!(G, A; prefactor=prefactor)
  return G
end

# Mutating softmax - vector input 
function softmax!(W::AbstractVector{Tf}, b::AbstractVector{Tf}; prefactor::Tf=Tf(1.0)) where {Tf<:AbstractFloat}
  @assert length(W) == length(b) "Length of W and b must be the same"
  if prefactor <= zero(Tf)
    throw(ArgumentError("prefactor must be positive"))
  end

  if length(b) == 0
    return nothing # W is also empty, nothing to do
  end

  # b is modified in place by this operation. 
  b ./= prefactor

  max_b = maximum(b)

  if max_b == Tf(-Inf)
    # All elements in b were -Inf or became -Inf after scaling.
    # Softmax is taken as uniform in this ambiguous case.
    W .= Tf(1.0) / length(b)
    return nothing
  end

  # Compute the exponentials
  @inbounds @simd for d in eachindex(b)
    W[d] = exp(b[d] - max_b)
  end

  # Normalise
  sum_W = sum(W)

  # Handle cases where sum_W is zero (all underflowed), negative (should not happen), NaN, or Inf.
  if sum_W <= eps(Tf) || !isfinite(sum_W)
    # Fallback to a uniform distribution.
    W .= Tf(1.0) / length(b)
  else
    # Standard normalisation
    @inbounds @simd for d in eachindex(W)
      W[d] /= sum_W
    end
  end
  
  return nothing
end

# Mutating softmax - vector input (W not provided)
function softmax!(b::AbstractVector{Tf}; prefactor::Tf=Tf(1.0)) where {Tf<:AbstractFloat}
  W = similar(b)
  softmax!(W, b; prefactor=prefactor)
  return W
end

"""
    softmax(A; prefactor=1.0)
    softmax(b; prefactor=1.0)

Computes the softmax function for a matrix or vector, returning a new array.

For matrix inputs `A`, the softmax is computed column-wise.
For vector inputs `b`, the softmax is computed over the vector elements.

The input array (`A` or `b`) is **not** modified. A copy is made internally
before scaling by `prefactor` and applying the softmax operation.

This function utilizes `softmax!` internally and thus benefits from its robust handling
of edge cases (empty inputs, `-Inf` values, problematic sums of exponentials).

# Arguments
- `A::AbstractMatrix{Tf}`: The input matrix.
- `b::AbstractVector{Tf}`: The input vector.

# Keyword Arguments
- `prefactor::Tf`: A positive scaling factor applied to a copy of the input array elements
  before the `exp` operation. Defaults to `Tf(1.0)`. An `ArgumentError` is thrown if `prefactor`
  is not positive.

# Returns
- `AbstractMatrix{Tf}`: A new matrix containing the column-wise softmax of `A`.
- `AbstractVector{Tf}`: A new vector containing the softmax of `b`.

# See Also
- [`softmax!`](@ref): Mutating version of this function.
"""
function softmax(A::AbstractMatrix{Tf}; prefactor::Tf=Tf(1.0)) where {Tf<:AbstractFloat}
  G = similar(A)
  # Pass a copy of A to softmax! to prevent modifying the original A
  softmax!(G, copy(A); prefactor=prefactor)
  return G
end

# Softmax - vector input 
function softmax(b::AbstractVector{Tf}; prefactor::Tf=Tf(1.0)) where {Tf<:AbstractFloat}
  W = similar(b)
  # Pass a copy of b to softmax! to prevent modifying the original b
  softmax!(W, copy(b); prefactor=prefactor)
  return W
end