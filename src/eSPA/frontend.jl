# Data front-end

# Get column names based on table access type:
_columnnames(X) = collect(_columnnames(X, Val(Tables.columnaccess(X))))
_columnnames(X, ::Val{true}) = Tables.columnnames(Tables.columns(X))
_columnnames(X, ::Val{false}) = Tables.columnnames(first(Tables.rows(X)))
# Column names for matrix input
function _columnnames(X::AbstractMatrix{Tf}) where {Tf<:AbstractFloat}
    return [Symbol("feature_$i") for i in axes(X, 1)]
end

# Reformat - no weights
function MMI.reformat(::eSPAClassifier, X, y)
    X_mat = MMI.matrix(X; transpose=true)
    y_int = MMI.int(y)
    classes = MMI.classes(y)
    # Extract dimensions and get Π
    Tf = eltype(X_mat)                                  # Floating point type
    M_classes = length(classes)                         # Total number of classes
    Pi_mat = get_pi(y_int, M_classes, Tf)               # Target matrix

    return (X_mat, Pi_mat, y_int, _columnnames(X), classes)
end

# Select rows - no weights
function MMI.selectrows(::eSPAClassifier, I, X_mat, Pi_mat, y_int, column_names, classes)
    return (view(X_mat, :, I), view(Pi_mat, :, I), view(y_int, I), column_names, classes)
end

# Reformat - with weights
function MMI.reformat(model::eSPAClassifier, X, y, w)
    X_mat, Pi_mat, y_int, column_names, classes = MMI.reformat(model, X, y)
    return (X_mat, Pi_mat, y_int, column_names, classes, w)
end

# Select rows - with weights
function MMI.selectrows(::eSPAClassifier, I, X_mat, Pi_mat, y_int, column_names, classes, w)
    return (
        view(X_mat, :, I),
        view(Pi_mat, :, I),
        view(y_int, I),
        column_names,
        classes,
        view(w, I),
    )
end

# Reformat - predict
function MMI.reformat(::eSPAClassifier, X)
    return (MMI.matrix(X; transpose=true),)
end

# Select rows - predict
function MMI.selectrows(::eSPAClassifier, I, X_mat)
    return (view(X_mat, :, I),)
end

# Helper function to check and format weights
function format_weights(w, y::AbstractVector{<:Integer}, Tf::Type{<:AbstractFloat}=Float64)
    w isa AbstractVector{<:Real} || throw(
        ArgumentError("Expected `weights === nothing` or `weights::AbstractVector{<:Real}"),
    )
    length(y) == length(w) || throw(
        ArgumentError("weights passed must have the same length as the target vector.")
    )
    weights = convert.(Tf, w)
    EntropicLearning.normalise!(weights)
    return weights
end

# Helper function to get Π from y_int
function get_pi(
    y_int::AbstractVector{<:Integer}, M_classes::Integer, Tf::Type{<:AbstractFloat}=Float64
)
    T_instances = length(y_int)
    Pi_mat = zeros(Tf, M_classes, T_instances)
    if T_instances > 0
        for t in 1:T_instances
            Pi_mat[y_int[t], t] = one(Tf)
        end
    end
    return Pi_mat
end
