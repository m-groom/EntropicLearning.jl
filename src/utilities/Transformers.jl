module Transformers

using MLJModelInterface
using Tables
using Statistics

const MMI = MLJModelInterface

export MinMaxScaler, QuantileTransformer

######### Constants ##########

const MIDPOINT_PROBABILITY = 0.5

######### Utility Functions ##########

function _validate_column_match(input_col_names, training_features)
    input_cols = Set(input_col_names)
    training_cols = Set(training_features)

    missing_cols = setdiff(training_cols, input_cols)
    extra_cols = setdiff(input_cols, training_cols)

    if !isempty(missing_cols) || !isempty(extra_cols)
        error_msg = "Column mismatch between input and training data. "
        if !isempty(missing_cols)
            error_msg *= "Missing columns: $(collect(missing_cols)). "
        end
        if !isempty(extra_cols)
            error_msg *= "Extra columns: $(collect(extra_cols)). "
        end
        error(error_msg)
    end
end

function _extract_column_vector(table, column_name)
    col_data_abstract = Tables.getcolumn(table, column_name)
    # Avoid collect if already an AbstractVector to reduce allocations
    return if col_data_abstract isa AbstractVector
        col_data_abstract
    else
        collect(col_data_abstract)
    end
end

function _create_feature_mapping(features)
    return Dict(feat => i for (i, feat) in enumerate(features))
end

function _build_named_tuple(column_names, column_vectors)
    # Convert to symbols if needed (handles both Symbol and String column names)
    sym_names = column_names isa AbstractVector{Symbol} ? column_names : Symbol.(column_names)
    return NamedTuple{Tuple(sym_names)}(Tuple(column_vectors))
end

function _get_promoted_eltype(table)
    col_names = Tables.columnnames(table)
    # Get all column types and promote them to find common supertype
    col_types = [Tables.columntype(table, name) for name in col_names]
    return promote_type(col_types...)
end

function _validate_feature_range(feature_range::Tuple{Float64,Float64})
    if feature_range[1] > feature_range[2]
        return "Upper bound of feature_range ($(feature_range[2])) must be greater than or equal to the lower bound ($(feature_range[1])). Resetting to (0.0, 1.0)."
    else
        return ""
    end
end

######### MinMaxScaler ##########

mutable struct MinMaxScaler <: MMI.Unsupervised
    feature_range::Tuple{Float64,Float64}
end

# Keyword constructor
function MinMaxScaler(; feature_range=(0.0, 1.0))
    transformer = MinMaxScaler(feature_range)
    message = MMI.clean!(transformer)
    isempty(message) || throw(ArgumentError(message))
    return transformer
end

function MMI.clean!(transformer::MinMaxScaler)
    err = _validate_feature_range(transformer.feature_range)
    if !isempty(err)
        transformer.feature_range = (0.0, 1.0)
    end
    return err
end

# Fit method: learns min and max for each feature
function MMI.fit(transformer::MinMaxScaler, verbosity::Int, X)
    # X is assumed to be a Tables.jl compatible table.
    col_names = Tables.columnnames(X)
    # Get promoted element type from all columns
    T = _get_promoted_eltype(X)
    # Pre-allocate result vectors with known size
    all_mins = Vector{T}(undef, length(col_names))
    all_maxs = Vector{T}(undef, length(col_names))

    for (col_idx, name) in enumerate(col_names)
        col_data = Tables.getcolumn(X, name)
        # Convert to an iterable collection if it's not already one (e.g. a generator) and
        # ensure elements are numbers.
        col_iterable = collect(T, col_data)
        if isempty(col_iterable)
            # Handle empty columns: use NaN
            all_mins[col_idx] = NaN
            all_maxs[col_idx] = NaN
        else
            all_mins[col_idx] = minimum(col_iterable)
            all_maxs[col_idx] = maximum(col_iterable)
        end
    end

    fitresult = (mins=all_mins, maxs=all_maxs, features=col_names)
    cache = nothing # No cache needed
    report = nothing

    return fitresult, cache, report
end

# transform method: applies the scaling
function MMI.transform(transformer::MinMaxScaler, fitresult, Xnew)
    col_names = Tables.columnnames(Xnew)
    data_mins = fitresult.mins
    data_maxs = fitresult.maxs
    features = fitresult.features

    # Validate that input columns exactly match training columns
    _validate_column_match(col_names, features)

    # Create mapping from feature name to index in training data
    feature_to_idx = _create_feature_mapping(features)

    # Get promoted element type from input table
    T = _get_promoted_eltype(Xnew)
    f_min, f_max = transformer.feature_range
    f_scale = f_max - f_min

    # Pre-allocate result vector with known size
    scaled_columns = Vector{AbstractVector{T}}(undef, length(col_names))

    for (col_idx, name) in enumerate(col_names)
        col_vector = _extract_column_vector(Xnew, name)

        # Use feature name to get correct min/max values
        feature_idx = feature_to_idx[name]
        current_data_min = data_mins[feature_idx]
        current_data_max = data_maxs[feature_idx]
        data_range = current_data_max - current_data_min

        scaled_col_vector = similar(col_vector, T)

        if data_range == 0.0
            # If data column is constant, map all values to f_min
            scaled_col_vector .= f_min
        else
            inv_data_range = 1.0 / data_range
            @inbounds @simd for i in eachindex(col_vector)
                # Standardise to [0,1] then scale to feature_range
                scaled_col_vector[i] =
                    (col_vector[i] - current_data_min) * inv_data_range * f_scale + f_min
            end
        end

        scaled_columns[col_idx] = scaled_col_vector
    end

    return _build_named_tuple(col_names, scaled_columns)
end

# inverse_transform method: reverses the scaling
function MMI.inverse_transform(transformer::MinMaxScaler, fitresult, Xscaled)
    col_names = Tables.columnnames(Xscaled)
    data_mins = fitresult.mins
    data_maxs = fitresult.maxs
    features = fitresult.features

    # Validate that input columns exactly match training columns
    _validate_column_match(col_names, features)

    # Create mapping from feature name to index in training data
    feature_to_idx = _create_feature_mapping(features)

    # Get promoted element type from input table
    T = _get_promoted_eltype(Xscaled)
    f_min, f_max = transformer.feature_range
    f_scale = f_max - f_min

    # Pre-allocate result vector with known size
    restored_columns = Vector{AbstractVector{T}}(undef, length(col_names))

    for (col_idx, name) in enumerate(col_names)
        scaled_col_vector = _extract_column_vector(Xscaled, name)

        # Use feature name to get correct min/max values
        feature_idx = feature_to_idx[name]
        current_data_min = data_mins[feature_idx]
        current_data_max = data_maxs[feature_idx]
        data_range = current_data_max - current_data_min

        restored_col_vector = similar(scaled_col_vector, T)
        if data_range == 0.0
            # If original data column was constant, all values should be current_data_min.
            restored_col_vector .= current_data_min
        elseif f_scale == 0.0
            # Original data had a range, but it was scaled to a single point (f_min).
            # All scaled values should ideally be f_min. The unscaled value (0-1 range) is 0
            restored_col_vector .= current_data_min # restore to current_data_min
        else
            # Both data_range and f_scale are non-zero.
            inv_f_scale = 1.0 / f_scale
            @inbounds @simd for i in eachindex(scaled_col_vector)
                # Ensure input to Float64 conversion if elements are not already floats
                val_01 = (scaled_col_vector[i] - f_min) * inv_f_scale
                restored_col_vector[i] = val_01 * data_range + current_data_min
            end
        end
        restored_columns[col_idx] = restored_col_vector
    end

    return _build_named_tuple(col_names, restored_columns)
end

# Fitted parameters
function MMI.fitted_params(::MinMaxScaler, fitresult)
    return (min_values_per_feature=fitresult.mins, max_values_per_feature=fitresult.maxs)
end

# MLJ Traits
MMI.metadata_model(
    MinMaxScaler;
    input_scitype=MMI.Table(MMI.Continuous),
    output_scitype=MMI.Table(MMI.Continuous),
    human_name="Min-Max Scaler",
    load_path="EntropicLearning.Transformers.MinMaxScaler",
)

######### QuantileTransformer ##########

mutable struct QuantileTransformer <: MMI.Unsupervised
    feature_range::Tuple{Float64,Float64}
end

# Keyword constructor
function QuantileTransformer(; feature_range=(0.0, 1.0))
    transformer = QuantileTransformer(feature_range)
    message = MMI.clean!(transformer)
    isempty(message) || throw(ArgumentError(message))
    return transformer
end

function MMI.clean!(transformer::QuantileTransformer)
    err = _validate_feature_range(transformer.feature_range)
    if !isempty(err)
        transformer.feature_range = (0.0, 1.0)
    end
    return err
end

function MMI.fit(transformer::QuantileTransformer, verbosity::Int, X)
    col_names = Tables.columnnames(X)
    # Get promoted element type from all columns
    T = _get_promoted_eltype(X)
    # Pre-allocate result vector with known size
    quantiles_per_column = Vector{Vector{T}}(undef, length(col_names))

    for (col_idx, name) in enumerate(col_names)
        col_data = Tables.getcolumn(X, name)
        # Convert to an iterable collection and ensure elements are numbers.
        col_iterable = if eltype(col_data) <: AbstractFloat
            collect(T, col_data)
        else
            T.(collect(col_data))
        end
        # Filter non-finite values
        numeric_col_data = filter(isfinite, col_iterable)

        if isempty(numeric_col_data)
            quantiles_per_column[col_idx] = T[] # Store empty if no valid data
        else
            quantiles_per_column[col_idx] = sort(unique(numeric_col_data))
        end
    end

    fitresult = (quantiles_list=quantiles_per_column, features=col_names)
    cache = nothing
    report = nothing

    return fitresult, cache, report
end

function MMI.transform(transformer::QuantileTransformer, fitresult, Xnew)
    Xnew_col_names = Tables.columnnames(Xnew)

    # Validate that input columns exactly match training columns
    _validate_column_match(Xnew_col_names, fitresult.features)

    # Create mapping from feature name to index in training data
    feature_to_idx = _create_feature_mapping(fitresult.features)

    # Get promoted element type from input table
    T = _get_promoted_eltype(Xnew)
    min_range, max_range = transformer.feature_range
    range_span = max_range - min_range

    # Pre-allocate result vector with known size
    transformed_cols = Vector{AbstractVector{T}}(undef, length(Xnew_col_names))

    for (col_idx, name) in enumerate(Xnew_col_names)
        col_vector = _extract_column_vector(Xnew, name)

        # Use feature name to get correct quantiles
        feature_idx = feature_to_idx[name]
        current_quantiles = fitresult.quantiles_list[feature_idx]
        n_quantiles = length(current_quantiles)

        new_col = similar(col_vector, T)

        if n_quantiles == 0
            fill!(new_col, (min_range + max_range) * MIDPOINT_PROBABILITY)
        elseif n_quantiles == 1
            q_val = current_quantiles[1]
            @inbounds for i in eachindex(col_vector)
                val = float(col_vector[i])
                p = if !isfinite(val)
                    MIDPOINT_PROBABILITY
                elseif val < q_val
                    0.0
                elseif val > q_val
                    1.0
                else # val == q_val
                    MIDPOINT_PROBABILITY # Convention for single quantile: map to midpoint
                end
                new_col[i] = p * range_span + min_range
            end
        else
            q_min = current_quantiles[1]
            q_max = current_quantiles[end]
            # Pre-calculate inverse of (n_quantiles - 1) to avoid repeated division
            inv_n_quantiles_minus_1 = 1.0 / (n_quantiles - 1)

            @inbounds for i in eachindex(col_vector)
                val = float(col_vector[i])
                p = 0.0

                if !isfinite(val)
                    p = MIDPOINT_PROBABILITY # Convention for non-finite values: map to midpoint of target range
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
                        q_i_plus_1 = current_quantiles[idx + 1]
                        p_i_plus_1 = idx * inv_n_quantiles_minus_1

                        denominator = q_i_plus_1 - q_i
                        fraction = denominator == 0.0 ? 0.0 : (val - q_i) / denominator
                        p = p_i + fraction * (p_i_plus_1 - p_i)
                    end
                end
                new_col[i] = p * range_span + min_range
            end
        end
        transformed_cols[col_idx] = new_col
    end

    # Reconstruct the table with the original column names from fitting
    output_col_names = fitresult.features
    if length(transformed_cols) != length(output_col_names)
        error(
            "Internal error: Number of transformed columns does not match number of " *
            "fitted column names.",
        )
    end

    return _build_named_tuple(output_col_names, transformed_cols)
end

function MMI.inverse_transform(transformer::QuantileTransformer, fitresult, Xtransformed)
    Xtransformed_col_names = Tables.columnnames(Xtransformed)

    # Validate that input columns exactly match training columns
    _validate_column_match(Xtransformed_col_names, fitresult.features)

    # Create mapping from feature name to index in training data
    feature_to_idx = _create_feature_mapping(fitresult.features)

    # Get promoted element type from input table
    T = _get_promoted_eltype(Xtransformed)
    min_range, max_range = transformer.feature_range
    range_span = max_range - min_range
    # Handle range_span == 0 separately to avoid division by zero with inv_range_span
    inv_range_span = range_span == 0.0 ? 0.0 : 1.0 / range_span # Will be used if range_span != 0

    # Pre-allocate result vector with known size
    original_cols = Vector{AbstractVector{T}}(undef, length(Xtransformed_col_names))

    for (col_idx, name) in enumerate(Xtransformed_col_names)
        col_vector = _extract_column_vector(Xtransformed, name)

        # Use feature name to get correct quantiles
        feature_idx = feature_to_idx[name]
        current_quantiles = fitresult.quantiles_list[feature_idx]
        n_quantiles = length(current_quantiles)

        new_col = similar(col_vector, T)
        if n_quantiles == 0
            fill!(new_col, NaN) # No quantiles, cannot determine original value
        elseif n_quantiles == 1
            fill!(new_col, current_quantiles[1]) # All values map to the single quantile
        else
            n_quantiles_minus_1 = n_quantiles - 1 # Cache this
            @inbounds for i in eachindex(col_vector)
                s_val = col_vector[i]
                p = 0.0

                if !isfinite(s_val)
                    new_col[i] = NaN
                    continue
                end

                if range_span == 0 # min_range == max_range: use MIDPOINT_PROBABILITY, implying the middle of the ECDF.
                    p = MIDPOINT_PROBABILITY
                else
                    p = (s_val - min_range) * inv_range_span
                end

                p = clamp(p, 0.0, 1.0) # Ensure p is within [0,1]

                # Interpolate based on p to find the original value from quantiles
                # idx_float is the fractional index into the quantiles array
                idx_float = p * n_quantiles_minus_1 + 1.0

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
        original_cols[col_idx] = new_col
    end

    output_col_names = fitresult.features
    if length(original_cols) != length(output_col_names)
        error(
            "Internal error: Number of inverse_transformed columns " *
            "does not match number of fitted column names.",
        )
    end

    return _build_named_tuple(output_col_names, original_cols)
end

# Fitted parameters
function MMI.fitted_params(::QuantileTransformer, fitresult)
    return (quantiles_list=fitresult.quantiles_list,)
end

# MLJ traits
MMI.metadata_model(
    QuantileTransformer;
    input_scitype=MMI.Table(MMI.Continuous),
    output_scitype=MMI.Table(MMI.Continuous),
    human_name="Quantile Transformer",
    load_path="EntropicLearning.Transformers.QuantileTransformer",
)

######### Documentation ##########

"""
$(MMI.doc_header(MinMaxScaler))

Use this model to scale features to a given range, defaulting to [0, 1]. Each feature
is scaled independently using a linear transformation based on the minimum and maximum
values observed during fitting. The rescalings applied by this transformer to new data
are always those learned during the training phase. The behaviour of this model is similar
to that of the `MinMaxScaler` in the `sklearn.preprocessing` Python package.


# Training data

In MLJ or MLJBase, bind an instance `model` to data with

    mach = machine(model, X)

where

- `X`: any Tables.jl compatible table or any abstract vector with
  `Continuous` element scitype (any abstract float vector). Only
  features in a table with `Continuous` scitype can be scaled;
  check column scitypes with `schema(X)`.

Train the machine using `fit!(mach, rows=...)`.


# Hyper-parameters

- `feature_range::Tuple{Float64, Float64}`: The desired range for the transformed data.
  Defaults to `(0.0, 1.0)`.


# Operations

- `transform(mach, Xnew)`: return `Xnew` with features scaled to the specified
  `feature_range` according to the min/max values learned during fitting of `mach`.
  The transformation formula is: `X_scaled = (X - data_min) / (data_max - data_min) * (range_max - range_min) + range_min`.
  If a feature has constant values (data_max == data_min), all values are mapped to `range_min`.

- `inverse_transform(mach, Z)`: apply the inverse transformation to `Z`, mapping
  values from `feature_range` back to the original feature domain using the
  min/max values learned during `fit`.


# Fitted parameters

The fields of `fitted_params(mach)` are:

- `min_values_per_feature` - the minimum values for each feature column learned during fitting

- `max_values_per_feature` - the maximum values for each feature column learned during fitting


# Examples

```
using MLJ

# Create example data with different scales
X = (a = [1.0, 2.0, 3.0, 4.0, 5.0],
     b = [5.0, 4.0, 3.0, 2.0, 1.0],
     c = [10.0, 20.0, 30.0, 40.0, 50.0])

julia> schema(X)
┌───────┬────────────┬─────────┐
│ names │ scitypes   │ types   │
├───────┼────────────┼─────────┤
│ a     │ Continuous │ Float64 │
│ b     │ Continuous │ Float64 │
│ c     │ Continuous │ Float64 │
└───────┴────────────┴─────────┘

# Default scaling to [0, 1]
scaler = MinMaxScaler()
mach = machine(scaler, X)
fit!(mach)

julia> X_transformed = transform(mach, X)
(a = [0.0, 0.25, 0.5, 0.75, 1.0],
 b = [1.0, 0.75, 0.5, 0.25, 0.0],
 c = [0.0, 0.25, 0.5, 0.75, 1.0],)

# Custom feature range
scaler2 = MinMaxScaler(feature_range=(-1.0, 1.0))
mach2 = machine(scaler2, X)
fit!(mach2)

julia> transform(mach2, X)
(a = [-1.0, -0.5, 0.0, 0.5, 1.0],
 b = [1.0, 0.5, 0.0, -0.5, -1.0],
 c = [-1.0, -0.5, 0.0, 0.5, 1.0],)

# Handling constant columns
X_const = (a = [1.0, 1.0, 1.0], b = [2.0, 3.0, 4.0])
mach_const = fit!(machine(scaler, X_const))

julia> transform(mach_const, X_const)
(a = [0.0, 0.0, 0.0],              # constant column mapped to range_min
 b = [0.0, 0.5, 1.0],)             # regular scaling

# Perfect inverse transformation
julia> X_restored = inverse_transform(mach, X_transformed)
(a = [1.0, 2.0, 3.0, 4.0, 5.0],
 b = [5.0, 4.0, 3.0, 2.0, 1.0],
 c = [10.0, 20.0, 30.0, 40.0, 50.0],)

# View fitted parameters
julia> fitted_params(mach)
(min_values_per_feature = [1.0, 1.0, 10.0],
 max_values_per_feature = [5.0, 5.0, 50.0],)
```

See also [`QuantileTransformer`](@ref).
"""
MinMaxScaler

"""
$(MMI.doc_header(QuantileTransformer))

Use this model to transform features to be uniformly distributed over a given range,
defaulting to [0, 1]. This transformation maps each feature to a uniform distribution
by calculating the empirical cumulative distribution function (ECDF) of the training
data. The rescalings applied by this transformer to new data are always those learned
during the training phase. The behaviour of this model is similar to that of the
`QuantileTransformer` in the `sklearn.preprocessing` Python package.


# Training data

In MLJ or MLJBase, bind an instance `model` to data with

    mach = machine(model, X)

where

- `X`: any Tables.jl compatible table or any abstract vector with
  `Continuous` element scitype (any abstract float vector). Only
  features in a table with `Continuous` scitype can be transformed;
  check column scitypes with `schema(X)`.

Train the machine using `fit!(mach, rows=...)`.


# Hyper-parameters

- `feature_range::Tuple{Float64, Float64}`: The desired range for the transformed data.
  Defaults to `(0.0, 1.0)`.


# Operations

- `transform(mach, Xnew)`: return `Xnew` with features transformed to uniform
  distribution over the specified `feature_range` according to the quantiles
  learned during fitting of `mach`. For out-of-sample values, those smaller than
  the training minimum are mapped to the lower bound of `feature_range`, and those
  larger than the training maximum are mapped to the upper bound. Interpolation is
  used for values falling between learned quantile values.

- `inverse_transform(mach, Z)`: apply the inverse transformation to `Z`, mapping
  values from `feature_range` back to the original feature domain using linear
  interpolation between the quantiles learned during `fit`.


# Fitted parameters

The fields of `fitted_params(mach)` are:

- `quantiles_list` - vector of quantile arrays, one for each feature column

- `col_names` - the names of features that were fitted


# Examples

```
using MLJ

# Create example data with different patterns
X = (a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
     b = [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
     c = [1.0, 1.0, 1.0, 5.0, 5.0, 5.0, 10.0, 10.0, 10.0, 10.0])

julia> schema(X)
┌───────┬────────────┬─────────┐
│ names │ scitypes   │ types   │
├───────┼────────────┼─────────┤
│ a     │ Continuous │ Float64 │
│ b     │ Continuous │ Float64 │
│ c     │ Continuous │ Float64 │
└───────┴────────────┴─────────┘

# Default transformation to [0, 1]
transformer = QuantileTransformer()
mach = machine(transformer, X)
fit!(mach)

julia> X_transformed = transform(mach, X)
(a = [0.0, 0.1111111111111111, 0.2222222222222222, 0.3333333333333333, 0.4444444444444444, 0.5555555555555556, 0.6666666666666666, 0.7777777777777777, 0.8888888888888888, 1.0],
 b = [1.0, 0.8888888888888888, 0.7777777777777777, 0.6666666666666666, 0.5555555555555556, 0.4444444444444444, 0.3333333333333333, 0.2222222222222222, 0.1111111111111111, 0.0],
 c = [0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0],)

# Custom feature range
transformer2 = QuantileTransformer(feature_range=(-1.0, 1.0))
mach2 = machine(transformer2, X)
fit!(mach2)

julia> transform(mach2, X)
(a = [-1.0, -0.7777777777777778, -0.5555555555555556, -0.33333333333333337, -0.11111111111111116, 0.11111111111111116, 0.33333333333333326, 0.5555555555555554, 0.7777777777777777, 1.0],
 b = [1.0, 0.7777777777777777, 0.5555555555555554, 0.33333333333333326, 0.11111111111111116, -0.11111111111111116, -0.33333333333333337, -0.5555555555555556, -0.7777777777777778, -1.0],
 c = [-1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],)

# Out-of-sample data handling
X_new = (a = [0.0, 5.5, 11.0],
         b = [12.0, 5.5, -1.0],
         c = [1.0, 7.0, 10.0])

julia> transform(mach, X_new)
(a = [0.0, 0.5, 1.0],
 b = [1.0, 0.5, 0.0],
 c = [0.0, 0.7, 1.0],)

# Inverse transformation
julia> X_restored = inverse_transform(mach, X_transformed)
(a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 7.999999999999999, 9.0, 10.0],
 b = [10.0, 9.0, 7.999999999999999, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
 c = [1.0, 1.0, 1.0, 5.0, 5.0, 5.0, 10.0, 10.0, 10.0, 10.0],)
```

See also [`MinMaxScaler`](@ref).
"""
QuantileTransformer

end # module Transformers
