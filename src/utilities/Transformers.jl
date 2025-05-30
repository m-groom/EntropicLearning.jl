module Transformers

using MLJBase
using Tables
using Statistics

export MinMaxScaler, QuantileTransformer

"""
    MinMaxScaler(; feature_range=(0.0, 1.0))

An unsupervised model for scaling features to a given range, defaulting to [0, 1].

For each feature (column) `X_col` in the input data, the transformation is:
1.  `X_std = (X_col - data_min) / (data_max - data_min)`
    (If `data_max == data_min`, `X_std` is 0 for all values in `X_col`)
2.  `X_scaled = X_std * (feature_range[2] - feature_range[1]) + feature_range[1]`

# Hyperparameters
- `feature_range::Tuple{Float64, Float64}`: The desired range for the transformed data.
  Defaults to `(0.0, 1.0)`.
"""
mutable struct MinMaxScaler <: MLJBase.Unsupervised
    feature_range::Tuple{Float64,Float64}
end

# Keyword constructor
function MinMaxScaler(; feature_range=(0.0, 1.0))
    if feature_range[1] > feature_range[2]
        error("Upper bound of feature_range ($(feature_range[2])) must be greater than or equal to the lower bound ($(feature_range[1])).")
    end
    return MinMaxScaler(feature_range)
end

# Fit method: learns min and max for each feature
function MLJBase.fit(transformer::MinMaxScaler, verbosity::Int, X)
    # X is assumed to be a Tables.jl compatible table.
    col_names = Tables.columnnames(X)
    all_mins = Float64[]
    all_maxs = Float64[]

    for name in col_names
        col_data = Tables.getcolumn(X, name)
        # Convert to an iterable collection if it's not already one (e.g. a generator) and ensure elements are numbers.
        col_iterable = collect(col_data)
        if isempty(col_iterable)
            # Handle empty columns: use NaN
            push!(all_mins, NaN)
            push!(all_maxs, NaN)
        else
            push!(all_mins, Float64(minimum(col_iterable)))
            push!(all_maxs, Float64(maximum(col_iterable)))
        end
    end

    fitresult = (mins=all_mins, maxs=all_maxs)
    cache = nothing # No cache needed
    report = nothing # TODO: return names of features that were scaled

    return fitresult, cache, report
end

# transform method: applies the scaling
function MLJBase.transform(transformer::MinMaxScaler, fitresult, X)
    col_names = Tables.columnnames(X)
    data_mins = fitresult.mins
    data_maxs = fitresult.maxs

    f_min, f_max = transformer.feature_range
    f_scale = f_max - f_min

    scaled_columns = Vector{AbstractVector{Float64}}()

    for (j, name) in enumerate(col_names)
        col_data_abstract = Tables.getcolumn(X, name)
        # Avoid collect if already an AbstractVector to reduce allocations
        col_vector = col_data_abstract isa AbstractVector ? col_data_abstract : collect(col_data_abstract)

        current_data_min = data_mins[j]
        current_data_max = data_maxs[j]
        data_range = current_data_max - current_data_min

        scaled_col_vector = similar(col_vector, Float64)

        if data_range == 0.0
            # If data column is constant, map all values to f_min
            scaled_col_vector .= f_min
        else
            inv_data_range = 1.0 / data_range
            for i in eachindex(col_vector)
                # Standardise to [0,1] then scale to feature_range
                scaled_col_vector[i] = (col_vector[i] - current_data_min) * inv_data_range * f_scale + f_min
            end
        end

        push!(scaled_columns, scaled_col_vector)
    end

    named_tuple_data = NamedTuple{Tuple(col_names)}(Tuple(scaled_columns))
    output_table = MLJBase.table(named_tuple_data)
    return output_table

end

# inverse_transform method: reverses the scaling
function MLJBase.inverse_transform(transformer::MinMaxScaler, fitresult, Xscaled)
    col_names = Tables.columnnames(Xscaled)
    data_mins = fitresult.mins
    data_maxs = fitresult.maxs

    f_min, f_max = transformer.feature_range
    f_scale = f_max - f_min

    restored_columns = Vector{AbstractVector{Float64}}()

    for (j, name) in enumerate(col_names)
        scaled_col_data_abstract = Tables.getcolumn(Xscaled, name)
        scaled_col_vector = scaled_col_data_abstract isa AbstractVector ? scaled_col_data_abstract : collect(scaled_col_data_abstract)

        current_data_min = data_mins[j]
        current_data_max = data_maxs[j]
        data_range = current_data_max - current_data_min

        restored_col_vector = similar(scaled_col_vector, Float64)
        if data_range == 0.0
            # If original data column was constant, all values should be current_data_min, regardless of f_scale.
            restored_col_vector .= current_data_min
        elseif f_scale == 0.0
            # Original data had a range, but it was scaled to a single point (f_min). All scaled values should ideally be f_min. The unscaled value (0-1 range) is 0. So, restore to current_data_min.
            restored_col_vector .= current_data_min
        else
            # Both data_range and f_scale are non-zero.
            inv_f_scale = 1.0 / f_scale
            for i in eachindex(scaled_col_vector)
                # Ensure input to Float64 conversion if elements are not already floats
                val_01 = (scaled_col_vector[i] - f_min) * inv_f_scale
                restored_col_vector[i] = val_01 * data_range + current_data_min
            end
        end
        push!(restored_columns, restored_col_vector)
    end

    named_tuple_data = NamedTuple{Tuple(col_names)}(Tuple(restored_columns))
    output_table = MLJBase.table(named_tuple_data)
    return output_table
end

# Specify input and output scitypes
MLJBase.input_scitype(::Type{<:MinMaxScaler}) = MLJBase.Table(MLJBase.Continuous)
MLJBase.output_scitype(::Type{<:MinMaxScaler}) = MLJBase.Table(MLJBase.Continuous)

# Fitted parameters
function MLJBase.fitted_params(::MinMaxScaler, fitresult) # TODO: also return names of features that were scaled
    return (min_values_per_feature=fitresult.mins, max_values_per_feature=fitresult.maxs)
end

"""
    QuantileTransformer(; feature_range=(0.0, 1.0))

An unsupervised model for scaling features to be uniformly distributed over a given range, defaulting to [0, 1].

For each feature (column) `X_col` in the input data, the transformation maps each feature to a uniform distribution by calculating the empirical cumulative distribution (ECDF) of the training data for that feature. 
Each value is then mapped to its ECDF value (percentile rank). 
These ranks (ranging from 0 to 1) are then linearly scaled to the specified `feature_range`.
For out-of-sample data, values smaller than the training minimum are mapped to the lower bound of `feature_range`, and values larger than the training maximum are mapped to the upper bound. 
Interpolation is used for values falling between learned quantile values.

The inverse transformation maps values from `feature_range` back to the original feature's domain using linear interpolation between the quantiles learned during `fit`.

# Hyperparameters
- `feature_range::Tuple{Float64, Float64}`: The desired range for the transformed data.
  Defaults to `(0.0, 1.0)`.
"""
mutable struct QuantileTransformer <: MLJBase.Unsupervised
    feature_range::Tuple{Float64,Float64}
end

# Keyword constructor
function QuantileTransformer(; feature_range=(0.0, 1.0))
    if feature_range[1] > feature_range[2]
        error("Upper bound of feature_range ($(feature_range[2])) must be greater than or equal to the lower bound ($(feature_range[1])).")
    end
    return QuantileTransformer(feature_range)
end

function MLJBase.fit(transformer::QuantileTransformer, verbosity::Int, X)
    col_names = Tables.columnnames(X)
    quantiles_per_column = Vector{Vector{Float64}}()

    for name in col_names
        col_data = Tables.getcolumn(X, name)
        # Convert to an iterable collection if it's not already one (e.g. a generator) and ensure elements are numbers.
        col_iterable = collect(eltype(col_data) <: AbstractFloat ? col_data : float.(col_data))
        # Filter non-finite values
        numeric_col_data = filter(isfinite, col_iterable)

        if isempty(numeric_col_data)
            push!(quantiles_per_column, Float64[]) # Store empty if no valid data
        else
            push!(quantiles_per_column, sort(unique(numeric_col_data)))
        end
    end

    fitresult = (quantiles_list=quantiles_per_column, col_names=col_names)
    cache = nothing
    report = nothing

    return fitresult, cache, report
end

function MLJBase.transform(transformer::QuantileTransformer, fitresult, Xnew)
    Xnew_col_names = Tables.columnnames(Xnew)
    if Xnew_col_names != fitresult.col_names
        error("Column names in Xnew do not match column names from fitting.")
    end

    min_range, max_range = transformer.feature_range
    range_span = max_range - min_range

    transformed_cols = Vector{AbstractVector{Float64}}()

    for (j, name) in enumerate(Xnew_col_names)
        col_data_abstract = Tables.getcolumn(Xnew, name)
        # Avoid collect if already an AbstractVector
        col_vector = col_data_abstract isa AbstractVector ? col_data_abstract : collect(col_data_abstract)

        # Ensure fitresult.quantiles_list has an entry for j
        if j > length(fitresult.quantiles_list)
            error("Mismatch in column count or order compared to fit data for column: $name")
        end
        current_quantiles = fitresult.quantiles_list[j]
        n_quantiles = length(current_quantiles)

        new_col = similar(col_vector, Float64)

        if n_quantiles == 0
            fill!(new_col, (min_range + max_range) * 0.5)
        elseif n_quantiles == 1
            q_val = current_quantiles[1]
            for i in eachindex(col_vector)
                val = float(col_vector[i])
                p = if !isfinite(val)
                    0.5
                elseif val < q_val
                    0.0
                elseif val > q_val
                    1.0
                else # val == q_val
                    0.5 # Convention for single quantile: map to midpoint
                end
                new_col[i] = p * range_span + min_range
            end
        else
            q_min = current_quantiles[1]
            q_max = current_quantiles[end]
            # Pre-calculate inverse of (n_quantiles - 1) to avoid repeated division
            inv_n_quantiles_minus_1 = 1.0 / (n_quantiles - 1)

            for i in eachindex(col_vector)
                val = float(col_vector[i])
                p = 0.0

                if !isfinite(val)
                    p = 0.5 # Convention for non-finite values: map to midpoint of target range
                elseif val <= q_min
                    p = 0.0
                elseif val >= q_max
                    p = 1.0
                else
                    # Find insertion point
                    idx = searchsortedlast(current_quantiles, val)
                    q_i = current_quantiles[idx]
                    p_i = (idx - 1) * inv_n_quantiles_minus_1

                    if val == q_i # Value falls exactly on a quantile
                        p = p_i
                    else # Interpolate between q_i and q_i_plus_1
                        q_i_plus_1 = current_quantiles[idx+1]
                        p_i_plus_1 = idx * inv_n_quantiles_minus_1

                        denominator = q_i_plus_1 - q_i
                        fraction = denominator == 0.0 ? 0.0 : (val - q_i) / denominator
                        p = p_i + fraction * (p_i_plus_1 - p_i)
                    end
                end
                new_col[i] = p * range_span + min_range
            end
        end
        push!(transformed_cols, new_col)
    end

    # Reconstruct the table with the original column names from fitting
    output_col_names = fitresult.col_names
    if length(transformed_cols) != length(output_col_names)
        error("Internal error: Number of transformed columns does not match number of fitted column names.")
    end

    named_tuple_data = NamedTuple{Tuple(Symbol.(output_col_names))}(Tuple(transformed_cols))
    output_table = MLJBase.table(named_tuple_data)
    return output_table
end

function MLJBase.inverse_transform(transformer::QuantileTransformer, fitresult, Xtransformed)
    Xtransformed_col_names = Tables.columnnames(Xtransformed)
    if Xtransformed_col_names != fitresult.col_names
        error("Column names in Xtransformed do not match column names from fitting.")
    end

    min_range, max_range = transformer.feature_range
    range_span = max_range - min_range
    # Handle range_span == 0 separately to avoid division by zero with inv_range_span
    inv_range_span = range_span == 0.0 ? 0.0 : 1.0 / range_span # Will be used if range_span != 0

    original_cols = Vector{AbstractVector{Float64}}()

    for (j, name) in enumerate(Xtransformed_col_names)
        col_data_abstract = Tables.getcolumn(Xtransformed, name)
        # Avoid collect if already an AbstractVector
        col_vector = col_data_abstract isa AbstractVector ? col_data_abstract : collect(col_data_abstract)

        if j > length(fitresult.quantiles_list)
            error("Mismatch in column count or order for inverse_transform for column: $name")
        end
        current_quantiles = fitresult.quantiles_list[j]
        n_quantiles = length(current_quantiles)

        new_col = similar(col_vector, Float64)
        if n_quantiles == 0
            fill!(new_col, NaN) # No quantiles, cannot determine original value
        elseif n_quantiles == 1
            fill!(new_col, current_quantiles[1]) # All values map to the single quantile
        else
            n_quantiles_minus_1 = n_quantiles - 1 # Cache this
            for i in eachindex(col_vector)
                s_val = col_vector[i]
                p = 0.0

                if !isfinite(s_val)
                    new_col[i] = NaN
                    continue
                end

                if range_span == 0 # min_range == max_range: use 0.5, implying the middle of the ECDF.
                    p = 0.5
                else
                    p = (s_val - min_range) * inv_range_span
                end

                p = clamp(p, 0.0, 1.0) # Ensure p is within [0,1]

                # Interpolate based on p to find the original value from quantiles
                idx_float = p * n_quantiles_minus_1 + 1.0   # idx_float is the fractional index into the quantiles array

                lower_idx = floor(Int, idx_float)
                upper_idx = ceil(Int, idx_float)

                # Clamp indices to be within bounds of current_quantiles array
                lower_idx = clamp(lower_idx, 1, n_quantiles)
                upper_idx = clamp(upper_idx, 1, n_quantiles)

                if lower_idx == upper_idx
                    new_col[i] = current_quantiles[lower_idx]
                else
                    weight = idx_float - lower_idx
                    val_lower = current_quantiles[lower_idx]
                    val_upper = current_quantiles[upper_idx]
                    new_col[i] = (1.0 - weight) * val_lower + weight * val_upper
                end
            end
        end
        push!(original_cols, new_col)
    end

    output_col_names = fitresult.col_names
    if length(original_cols) != length(output_col_names)
        error("Internal error: Number of inverse_transformed columns does not match number of fitted column names.")
    end

    named_tuple_data = NamedTuple{Tuple(Symbol.(output_col_names))}(Tuple(original_cols))
    output_table = MLJBase.table(named_tuple_data)
    return output_table
end

# MLJ traits
MLJBase.input_scitype(::Type{<:QuantileTransformer}) = MLJBase.Table(MLJBase.Continuous)
MLJBase.target_scitype(::Type{<:QuantileTransformer}) = MLJBase.Table(MLJBase.Continuous) # Output of transform

function MLJBase.fitted_params(::QuantileTransformer, fitresult)
    return (quantiles_list=fitresult.quantiles_list, col_names=fitresult.col_names)
end

end # module Transformers 