using Test
using EntropicLearning # Or using .Transformers if running from within EntropicLearning.jl package context
using MLJBase
using Tables
using Statistics

# Test data
X_table = (
    a=[1.0, 2.0, 3.0, 4.0, 5.0],
    b=[5.0, 4.0, 3.0, 2.0, 1.0],
    c=[10.0, 20.0, 30.0, 40.0, 50.0]
)
X = MLJBase.table(X_table)

# MinMaxScaler tests
@testset "MinMaxScaler Tests" begin
    @testset "Default feature_range (0, 1)" begin
        scaler = MinMaxScaler()
        mach = machine(scaler, X)
        fit!(mach)
        X_transformed = transform(mach, X)

        # Check mins and maxs of transformed data
        for col_name in Tables.columnnames(X_transformed)
            col_data = Tables.getcolumn(X_transformed, col_name)
            @test minimum(col_data) ≈ 0.0 atol=1e-9
            @test maximum(col_data) ≈ 1.0 atol=1e-9
        end

        # Check inverse_transform
        X_restored = inverse_transform(mach, X_transformed)
        for col_name in Tables.columnnames(X)
            original_col = Tables.getcolumn(X, col_name)
            restored_col = Tables.getcolumn(X_restored, col_name)
            @test original_col ≈ restored_col atol=1e-9
        end
    end

    @testset "Custom feature_range (-1, 1)" begin
        fmin, fmax = -1.0, 1.0
        scaler = MinMaxScaler(feature_range=(fmin, fmax))
        mach = machine(scaler, X)
        fit!(mach)
        X_transformed = transform(mach, X)

        for col_name in Tables.columnnames(X_transformed)
            col_data = Tables.getcolumn(X_transformed, col_name)
            @test minimum(col_data) ≈ fmin atol=1e-9
            @test maximum(col_data) ≈ fmax atol=1e-9
        end

        # Inverse transform
        X_restored = inverse_transform(mach, X_transformed)
        for col_name in Tables.columnnames(X)
            original_col = Tables.getcolumn(X, col_name)
            restored_col = Tables.getcolumn(X_restored, col_name)
            @test original_col ≈ restored_col atol=1e-9
        end
    end

    @testset "Constant column" begin
        X_const_table = (a=[1.0, 1.0, 1.0], b=[2.0, 3.0, 4.0])
        X_const = MLJBase.table(X_const_table)
        fmin, fmax = 0.0, 1.0
        scaler = MinMaxScaler(feature_range=(fmin, fmax))
        mach = machine(scaler, X_const)
        fit!(mach)
        X_transformed = transform(mach, X_const)

        a_transformed = Tables.getcolumn(X_transformed, :a)
        @test all(a_transformed .≈ fmin) 

        b_transformed = Tables.getcolumn(X_transformed, :b)
        @test minimum(b_transformed) ≈ fmin atol=1e-9
        @test maximum(b_transformed) ≈ fmax atol=1e-9

        # Inverse transform
        X_restored = inverse_transform(mach, X_transformed)
        for col_name in Tables.columnnames(X_const)
            original_col = Tables.getcolumn(X_const, col_name)
            restored_col = Tables.getcolumn(X_restored, col_name)
            @test original_col ≈ restored_col atol=1e-9
        end
    end

    @testset "Feature range with zero width" begin
        fmin, fmax = 3.0, 3.0
        scaler = MinMaxScaler(feature_range=(fmin, fmax))
        mach = machine(scaler, X)
        fit!(mach)
        X_transformed = transform(mach, X)

        for col_name in Tables.columnnames(X_transformed)
            col_data = Tables.getcolumn(X_transformed, col_name)
            @test all(col_data .≈ fmin) 
        end
    end

    @testset "Error for invalid feature_range" begin
        @test_throws ErrorException MinMaxScaler(feature_range=(1.0, 0.0))
    end

    @testset "Empty column" begin
        X_empty_col_table = (a=[1.0, 2.0], b=Float64[])
        X_empty_col = MLJBase.table(X_empty_col_table)
        scaler = MinMaxScaler()
        mach = machine(scaler, X_empty_col)
        fit!(mach)
        fp = fitted_params(mach)
        @test isnan(fp.min_values_per_feature[2])
        @test isnan(fp.max_values_per_feature[2])
        X_transformed = transform(mach, X_empty_col)
        b_transformed = Tables.getcolumn(X_transformed, :b)
        @test all(isnan, b_transformed) # Behavior for empty columns might need to be explicitly defined if NaN is not desired.

        # Test inverse transform with NaN fitresults
        X_restored = inverse_transform(mach, X_transformed)
        b_restored = Tables.getcolumn(X_restored, :b)
        @test all(isnan, b_restored)
    end

end

# QuantileTransformer tests
@testset "QuantileTransformer Tests" begin
    # Test data for QuantileTransformer
    X_qt_table = (
        a=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        b=[10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
        c=[1.0, 1.0, 1.0, 5.0, 5.0, 5.0, 10.0, 10.0, 10.0, 10.0] # Includes ties
    )
    X_qt = MLJBase.table(X_qt_table)

    @testset "Default feature_range (0, 1)" begin
        transformer = QuantileTransformer()
        mach = machine(transformer, X_qt)
        fit!(mach)
        X_transformed = transform(mach, X_qt)

        for col_name in Tables.columnnames(X_transformed)
            col_data = Tables.getcolumn(X_transformed, col_name)
            @test minimum(col_data) >= 0.0 - 1e-9
            @test maximum(col_data) <= 1.0 + 1e-9
            # Check if ranks are somewhat uniformly distributed (harder to test strictly than min/max)
            # For unique values, ranks should be (i-1)/(n-1)
            # For tied values, interpolation occurs.
            if col_name == :a # Unique values
                expected_ranks = (0:9) ./ 9.0
                @test col_data ≈ expected_ranks atol=1e-9
            end
            if col_name == :b # Unique values (reversed)
                expected_ranks = reverse((0:9) ./ 9.0)
                @test col_data ≈ expected_ranks atol=1e-9
            end
        end

        # Check inverse_transform
        X_restored = inverse_transform(mach, X_transformed)
        for col_name in Tables.columnnames(X_qt)
            original_col = Tables.getcolumn(X_qt, col_name)
            restored_col = Tables.getcolumn(X_restored, col_name)
            @test original_col ≈ restored_col atol=1e-9
        end
    end

    @testset "Custom feature_range (-1, 1)" begin
        fmin, fmax = -1.0, 1.0
        transformer = QuantileTransformer(feature_range=(fmin, fmax))
        mach = machine(transformer, X_qt)
        fit!(mach)
        X_transformed = transform(mach, X_qt)

        for col_name in Tables.columnnames(X_transformed)
            col_data = Tables.getcolumn(X_transformed, col_name)
            @test minimum(col_data) >= fmin - 1e-9
            @test maximum(col_data) <= fmax + 1e-9
            if col_name == :a # Unique values
                expected_ranks_01 = (0:9) ./ 9.0
                expected_ranks_custom = expected_ranks_01 .* (fmax - fmin) .+ fmin
                @test col_data ≈ expected_ranks_custom atol=1e-9
            end
        end

        X_restored = inverse_transform(mach, X_transformed)
        for col_name in Tables.columnnames(X_qt)
            original_col = Tables.getcolumn(X_qt, col_name)
            restored_col = Tables.getcolumn(X_restored, col_name)
            @test original_col ≈ restored_col atol=1e-9
        end
    end

    @testset "Constant column" begin
        X_const_table = (a=[1.0, 1.0, 1.0], b=[2.0, 3.0, 4.0])
        X_const = MLJBase.table(X_const_table)
        fmin, fmax = 0.0, 1.0
        transformer = QuantileTransformer(feature_range=(fmin, fmax))
        mach = machine(transformer, X_const)
        fit!(mach)
        X_transformed = transform(mach, X_const)

        a_transformed = Tables.getcolumn(X_transformed, :a)
        # For a constant column, all ranks should be 0.5 (mid-point), then scaled to feature_range
        @test all(a_transformed .≈ 0.5 * (fmax - fmin) + fmin) 

        b_transformed = Tables.getcolumn(X_transformed, :b)
        expected_b_ranks = [0.0, 0.5, 1.0]
        @test b_transformed ≈ expected_b_ranks atol=1e-9

        X_restored = inverse_transform(mach, X_transformed)
        for col_name in Tables.columnnames(X_const)
            original_col = Tables.getcolumn(X_const, col_name)
            restored_col = Tables.getcolumn(X_restored, col_name)
            @test original_col ≈ restored_col atol=1e-9
        end
    end

    @testset "Out-of-sample data" begin
        transformer = QuantileTransformer()
        mach = machine(transformer, X_qt)
        fit!(mach)

        X_new_table = (a=[0.0, 5.5, 11.0], b=[12.0, 5.5, -1.0], c=[1.0, 7.0, 10.0])
        X_new = MLJBase.table(X_new_table)
        X_transformed = transform(mach, X_new)

        # Values outside fitted range should be clamped to feature_range bounds
        @test Tables.getcolumn(X_transformed, :a)[1] ≈ 0.0 atol=1e-9 # 0.0 < min(X_qt.a)
        @test Tables.getcolumn(X_transformed, :a)[3] ≈ 1.0 atol=1e-9 # 11.0 > max(X_qt.a)
        @test Tables.getcolumn(X_transformed, :b)[1] ≈ 1.0 atol=1e-9 # 12.0 > max(X_qt.b) -> smallest rank, maps to 1 for reversed
        @test Tables.getcolumn(X_transformed, :b)[3] ≈ 0.0 atol=1e-9 # -1.0 < min(X_qt.b) -> largest rank, maps to 0 for reversed

        # Inverse transform for out-of-sample (especially clamped values)
        # This tests if values transformed to 0 or 1 are correctly mapped back to the min/max of the *original* fitted quantiles
        X_restored_new = inverse_transform(mach, X_transformed)
        fit_params = fitted_params(mach)

        @test Tables.getcolumn(X_restored_new, :a)[1] ≈ fit_params.quantiles_list[1][1] atol=1e-9   # Mapped to min of fitted quantiles for 'a'
        @test Tables.getcolumn(X_restored_new, :a)[3] ≈ fit_params.quantiles_list[1][end] atol=1e-9 # Mapped to max
        @test Tables.getcolumn(X_restored_new, :b)[3] ≈ fit_params.quantiles_list[2][1] atol=1e-9   # Mapped to min (which is largest original value for 'b')
        @test Tables.getcolumn(X_restored_new, :b)[1] ≈ fit_params.quantiles_list[2][end] atol=1e-9 # Mapped to max (smallest original for 'b')
    end

    @testset "Error for invalid feature_range" begin
        @test_throws ErrorException QuantileTransformer(feature_range=(1.0, 0.0))
    end

    @testset "Empty column in fit" begin
        X_empty_fit_table = (a=[1.0, 2.0], b=Float64[])
        X_empty_fit = MLJBase.table(X_empty_fit_table)
        transformer = QuantileTransformer()
        mach = machine(transformer, X_empty_fit)
        fit!(mach)
        fp = fitted_params(mach)
        @test isempty(fp.quantiles_list[2])

        X_new_table = (a=[1.5], b=[10.0])
        X_new = MLJBase.table(X_new_table)
        X_transformed = transform(mach, X_new)
        # For column 'b' (empty during fit), transform should map to middle of feature_range
        @test Tables.getcolumn(X_transformed, :b)[1] ≈ 0.5 atol=1e-9

        # Inverse transform of such a column should result in NaN as per current implementation
        X_restored = inverse_transform(mach, X_transformed)
        @test isnan(Tables.getcolumn(X_restored, :b)[1])
    end

    @testset "All non-finite column in fit" begin
        X_nonfinite_table = (a=[1.0, 2.0], b=[NaN, Inf, -Inf])
        X_nonfinite = MLJBase.table(X_nonfinite_table)
        transformer = QuantileTransformer()
        mach = machine(transformer, X_nonfinite)
        fit!(mach)
        fp = fitted_params(mach)
        @test isempty(fp.quantiles_list[2]) # No finite values to compute quantiles

        X_new_table = (a=[1.5], b=[10.0])
        X_new = MLJBase.table(X_new_table)
        X_transformed = transform(mach, X_new)
        @test Tables.getcolumn(X_transformed, :b)[1] ≈ 0.5 atol=1e-9 # Middle of range
    end

    @testset "Column name mismatch" begin
        transformer = QuantileTransformer()
        mach = machine(transformer, X_qt)
        fit!(mach)
        X_wrong_names_table = (x1=X_qt.a, x2=X_qt.b, x3=X_qt.c)
        X_wrong_names = MLJBase.table(X_wrong_names_table)
        @test_throws ErrorException transform(mach, X_wrong_names)

        X_transformed_correct = transform(mach, X_qt)
        # Create a table with transformed data but wrong names for inverse_transform
        X_transformed_wrong_names_table = (x1=Tables.getcolumn(X_transformed_correct, :a), x2=Tables.getcolumn(X_transformed_correct, :b), x3=Tables.getcolumn(X_transformed_correct, :c))
        X_transformed_wrong_names = MLJBase.table(X_transformed_wrong_names_table)
        @test_throws ErrorException inverse_transform(mach, X_transformed_wrong_names)
    end

end