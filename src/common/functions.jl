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