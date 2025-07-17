using Test
using EntropicLearning
using MLJBase
using Random
using Statistics
using LinearAlgebra
using MLJModelInterface

# Test data helper function
function create_test_distances(n=10; outlier_ratio=0.1)
    Random.seed!(123)
    n_outliers = round(Int, n * outlier_ratio)
    n_normal = n - n_outliers

    # Normal samples have small distances
    normal_distances = rand(n_normal) * 0.1
    # Outliers have large distances
    outlier_distances = rand(n_outliers) * 2.0 .+ 1.0

    return vcat(normal_distances, outlier_distances)
end

@testset "EOS Utilities" begin
    @testset "eos_weights function" begin
        @testset "Basic functionality" begin
            distances = [0.1, 0.2, 2.0, 0.1, 0.3]
            alpha = 1.0
            weights = EntropicLearning.eos_weights(distances, alpha)

            @test length(weights) == length(distances)
            @test all(weights .>= 0)
            @test isapprox(sum(weights), 1.0, atol=1e-10)
            @test eltype(weights) <: AbstractFloat

            # Check that larger distances get smaller weights
            largest_distance_idx = argmax(distances)
            @test weights[largest_distance_idx] < mean(weights)
        end

        @testset "Edge cases" begin
            # Empty distances
            @test EntropicLearning.eos_weights(Float64[], 1.0) == Float64[]

            # Single distance
            weights = EntropicLearning.eos_weights([1.0], 1.0)
            @test length(weights) == 1
            @test isapprox(weights[1], 1.0, atol=1e-10)

            # All equal distances
            weights = EntropicLearning.eos_weights([1.0, 1.0, 1.0], 1.0)
            @test all(isapprox.(weights, 1/3, atol=1e-10))
        end

        @testset "Alpha parameter effects" begin
            distances = create_test_distances(20)

            # Small alpha → more concentrated weights
            weights_small = EntropicLearning.eos_weights(distances, 0.1)
            # Large alpha → more uniform weights
            weights_large = EntropicLearning.eos_weights(distances, 10.0)

            # Check that larger alpha leads to more uniform distribution
            entropy_small = EntropicLearning.entropy(weights_small)
            entropy_large = EntropicLearning.entropy(weights_large)
            @test entropy_small < entropy_large
        end

        @testset "Infinite alpha" begin
            distances = create_test_distances(10)
            weights = EntropicLearning.eos_weights(distances, Inf)

            # Should be uniform distribution
            @test all(isapprox.(weights, 1/length(distances), atol=1e-10))
        end

        @testset "Error handling" begin
            @test_throws ErrorException EntropicLearning.eos_weights([1.0, 2.0], 0.0)  # alpha = 0
            @test_throws ErrorException EntropicLearning.eos_weights([1.0, 2.0], -1.0)  # negative alpha
        end

        @testset "Type stability" begin
            # Test with different numeric types
            distances_int = [1, 2, 3]
            weights_int = EntropicLearning.eos_weights(distances_int, 1.0)
            @test eltype(weights_int) <: AbstractFloat

            distances_float32 = Float32[1.0, 2.0, 3.0]
            weights_float32 = EntropicLearning.eos_weights(distances_float32, 1.0)
            @test eltype(weights_float32) == Float32
        end
    end

    @testset "eos_weights with target effective dimension" begin
        distances = create_test_distances(100)

        @testset "Basic functionality" begin
            target_Deff = 0.5  # Effective sample size of T/2
            alpha_range = (0.01, 10.0)

            result = EntropicLearning.eos_weights(distances, alpha_range, target_Deff)
            weights, found_alpha = result.weights, result.alpha

            @test length(weights) == length(distances)
            @test all(weights .>= 0)
            @test isapprox(sum(weights), 1.0, atol=1e-10)
            @test alpha_range[1] <= found_alpha <= alpha_range[2]

            # Check that effective dimension is close to target
            actual_Deff = EntropicLearning.effective_dimension(weights; normalise=true)
            @test isapprox(actual_Deff, target_Deff, atol=1e-5)
        end

        @testset "Error handling" begin
            # Invalid alpha range
            @test_throws ErrorException EntropicLearning.eos_weights(
                distances, (0.0, 1.0), 0.5
            )  # alpha range contains 0
            @test_throws AssertionError EntropicLearning.eos_weights(
                distances, (2.0, 1.0), 0.5
            )  # invalid range
        end
    end

    @testset "calculate_eos_weights function" begin
        # Create test model and data
        X, y = make_blobs(100, 3; centers=3, cluster_std=1.0, rng=123)
        model = eSPAClassifier(K=3)
        mach = fit!(machine(model, X, y), verbosity=0)

        @testset "Basic functionality" begin
            alpha = 1.0
            weights = calculate_eos_weights(model, mach.fitresult, alpha, X, y)

            @test length(weights) == nrows(X)
            @test all(weights .>= 0)
            @test isapprox(sum(weights), 1.0, atol=1e-10)
        end

        @testset "With target effective dimension" begin
            alpha_range = (0.1, 10.0)
            target_Deff = 0.6

            result = calculate_eos_weights(model, mach.fitresult, alpha_range, target_Deff, X, y)
            weights, found_alpha = result.weights, result.alpha

            @test length(weights) == nrows(X)
            @test all(weights .>= 0)
            @test isapprox(sum(weights), 1.0, atol=1e-10)
            @test alpha_range[1] <= found_alpha <= alpha_range[2]
        end
    end

    @testset "eos_outlier_scores function" begin
        # Create test model and data
        X, y = make_blobs(100, 3; centers=3, cluster_std=1.0, rng=123)
        model = eSPAClassifier(K=3)
        mach = fit!(machine(model, X, y), verbosity=0)

        @testset "Basic functionality" begin
            alpha = 1.0
            scores = eos_outlier_scores(model, mach.fitresult, alpha, X, y)

            @test length(scores) == nrows(X)
            @test all(0 .<= scores .<= 1)
        end

        @testset "With target effective dimension" begin
            alpha_range = (0.1, 10.0)
            target_Deff = 0.6

            result = eos_outlier_scores(model, mach.fitresult, alpha_range, target_Deff, X, y)
            scores, found_alpha = result.scores, result.alpha

            @test length(scores) == nrows(X)
            @test all(0 .<= scores .<= 1)
            @test alpha_range[1] <= found_alpha <= alpha_range[2]
        end
    end

    @testset "eos_distances protocol" begin
        @testset "Default error behavior" begin
            struct DummyModel <: MLJBase.Unsupervised end
            dummy_model = DummyModel()

            @test_throws ErrorException eos_distances(dummy_model, nothing, [1, 2, 3])
        end

        @testset "eSPAClassifier implementation" begin
            X, y = make_blobs(50, 3; centers=3, cluster_std=1.0, rng=123)
            model = eSPAClassifier(K=3)
            mach = fit!(machine(model, X, y), verbosity=0)

            # Test with y provided (training case)
            args = MLJModelInterface.reformat(model, X, y)
            distances_train = eos_distances(model, mach.fitresult, args...)
            @test length(distances_train) == nrows(X)
            @test all(distances_train .>= 0)

            # Test without y (predict case)
            args = MLJModelInterface.reformat(model, X)
            distances_predict = eos_distances(model, mach.fitresult, args...)
            @test length(distances_predict) == nrows(X)
            @test all(distances_predict .>= 0)
        end
    end
end

@testset "EOS Wrapper" begin

    # Test data generation
    D_features = 3
    T_instances = 100
    K_clusters = 3
    # Data in MLJ format
    X_table, y_cat = MLJBase.make_blobs(
        T_instances, D_features; centers=K_clusters, cluster_std=1.0, rng=123, as_table=true
    )
    model = eSPAClassifier(K=K_clusters, epsC=1e-2, epsW=1e-1, random_state=101, max_iter=1)
    eos_model = EOSWrapper(model; alpha=0.1, max_iter=100, tol=1e-6)
    mach = machine(eos_model, X_table, y_cat)
    fit!(mach, verbosity=0)
    rep = MLJBase.report(mach)

    @testset "Constructor and validation" begin

        @testset "Valid construction" begin
            # Positional argument
            eos1 = EOSWrapper(eSPAClassifier(K=5), alpha=0.1, tol=1e-6, max_iter=200)
            @test eos1.model isa eSPAClassifier
            @test eos1.alpha == 0.1
            @test eos1.tol == 1e-6
            @test eos1.max_iter == 200

            # Keyword argument
            eos2 = EOSWrapper(; model=eSPAClassifier(K=5), alpha=2.0, tol=1e-5, max_iter=50)
            @test eos2.model isa eSPAClassifier
            @test eos2.alpha == 2.0
            @test eos2.tol == 1e-5
            @test eos2.max_iter == 50
        end

        @testset "Error handling" begin
            test_model = eSPAClassifier(K=5)

            # Invalid parameters should now warn and reset
            @test_warn "alpha must be positive" EOSWrapper(test_model; alpha=0.0)
            @test_warn "tol must be positive" EOSWrapper(test_model; tol=0.0)
            @test_warn "max_iter must be positive" EOSWrapper(test_model; max_iter=0)
            @test_warn "atol must be positive" EOSWrapper(test_model; atol=0.0)

            # Too many positional arguments
            @test_throws ArgumentError EOSWrapper(test_model, test_model)

            # No model provided
            @test_throws ArgumentError EOSWrapper(; alpha=1.0)
        end

        @testset "Wrapper type selection" begin
            # Test probabilistic wrapper for eSPAClassifier
            eos = EOSWrapper(eSPAClassifier(K=5))
            @test eos isa EntropicLearning.EOS.ProbabilisticEOSWrapper
        end
    end

    @testset "MLJ integration" begin

        @testset "Basic fitting and prediction" begin
            # Check fitresult structure
            @test mach.fitresult isa EntropicLearning.EOS.EOSFitResult
            weights = MLJModelInterface.fitted_params(mach).weights
            @test length(weights) == nrows(X_table)
            @test all(weights .>= 0)
            @test isapprox(sum(weights), 1.0, atol=1e-10)

            # Check report
            @test haskey(rep, :iterations)
            @test haskey(rep, :loss)
            @test haskey(rep, :ESS)
            @test haskey(rep, :timings)

            # Test prediction
            X_test, _ = make_blobs(10, D_features; centers=K_clusters, cluster_std=1.0, rng=456)
            ŷ = predict(mach, X_test)
            @test length(ŷ) == nrows(X_test)

            # Test transform
            scores = transform(mach, X_test)
            @test length(scores) == nrows(X_test)
            @test all(0 .<= scores .<= 1)
        end

        @testset "Convergence behavior" begin
            # Check that loss is decreasing
            losses = rep.loss
            if length(losses) > 1
                # Allow for small numerical increases
                for i in 2:length(losses)
                    @test losses[i] - losses[i-1] <= 1e-10
                end
            end
        end

        @testset "Fitted parameters" begin
            fitted_params = MLJBase.fitted_params(mach)

            @test haskey(fitted_params, :weights)
            @test haskey(fitted_params, :inner_params)
            @test length(fitted_params.weights) == nrows(X_table)
        end

        @testset "Different alpha values" begin
            # Test with small alpha (more concentrated weights)
            eos_small = EOSWrapper(model; alpha=0.1, max_iter=5)
            mach_small = fit!(machine(eos_small, X_table, y_cat), verbosity=0)

            # Test with large alpha (more uniform weights)
            eos_large = EOSWrapper(model; alpha=10.0, max_iter=5)
            mach_large = fit!(machine(eos_large, X_table, y_cat), verbosity=0)

            # Large alpha should lead to more uniform weights (higher entropy)
            entropy_small = EntropicLearning.entropy(MLJModelInterface.fitted_params(mach_small).weights)
            entropy_large = EntropicLearning.entropy(MLJModelInterface.fitted_params(mach_large).weights)

            @test entropy_small < entropy_large
        end
    end

    @testset "Outlier detection capability" begin
        # Create clean data
        X_clean, y_clean = make_blobs(80, D_features; centers=K_clusters, cluster_std=0.5, rng=123, as_table=false)

        # Add obvious outliers
        X_outliers = randn(20, D_features) * 5  # Large values, far from clusters
        y_outliers = rand(1:3, 20)     # Random class labels

        X = MLJBase.table(vcat(X_clean, X_outliers))
        y = MLJBase.categorical(vcat(y_clean, y_outliers))

        # Fit EOS model
        mach = fit!(machine(eos_model, X, y), verbosity=0)

        # Get outlier scores
        weights = MLJBase.fitted_params(mach).weights
        scores = EntropicLearning.outlier_scores(weights)

        # Outliers should have higher scores
        clean_scores = scores[1:80]
        outlier_scores = scores[81:100]

        @test mean(outlier_scores) > mean(clean_scores)
        @test maximum(outlier_scores) > 0.5  # At least some outliers should be clearly identified
    end

    @testset "Model compatibility check" begin
        # Test that EOSWrapper rejects models that don't support weights
        struct DummyUnsupportedModel end
        MLJModelInterface.is_supervised(::Type{DummyUnsupportedModel}) = true
        MLJModelInterface.supports_weights(::Type{DummyUnsupportedModel}) = false

        dummy_model = DummyUnsupportedModel()
        @test_throws ArgumentError EOSWrapper(dummy_model)
    end
end
