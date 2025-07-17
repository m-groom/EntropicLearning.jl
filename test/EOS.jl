using Test
using EntropicLearning
using MLJBase
using Random
using Statistics
using LinearAlgebra
using MLJModelInterface
using TimerOutputs

# Access EOS module
import EntropicLearning.EOS as EOS

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

            result = calculate_eos_weights(
                model, mach.fitresult, alpha_range, target_Deff, X, y
            )
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

            result = eos_outlier_scores(
                model, mach.fitresult, alpha_range, target_Deff, X, y
            )
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
            X_test, _ = make_blobs(
                10, D_features; centers=K_clusters, cluster_std=1.0, rng=456
            )
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
                    @test losses[i] - losses[i - 1] <= 1e-10
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
            entropy_small = EntropicLearning.entropy(
                MLJModelInterface.fitted_params(mach_small).weights
            )
            entropy_large = EntropicLearning.entropy(
                MLJModelInterface.fitted_params(mach_large).weights
            )

            @test entropy_small < entropy_large
        end
    end

    @testset "Outlier detection capability" begin
        # Create clean data
        X_clean, y_clean = make_blobs(
            80, D_features; centers=K_clusters, cluster_std=0.5, rng=123, as_table=false
        )

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

@testset "1. EOS Core Algorithm Tests" begin
    # Test data generation
    D_features = 3
    T_instances = 50  # Smaller for faster tests
    K_clusters = 3
    X_table, y_cat = MLJBase.make_blobs(
        T_instances, D_features; centers=K_clusters, cluster_std=1.0, rng=123, as_table=true
    )

    # Create inner model and EOS wrapper
    inner_model = eSPAClassifier(
        K=K_clusters, epsC=1e-2, epsW=1e-1, random_state=101, max_iter=1
    )
    eos_model = EOSWrapper(inner_model; alpha=1.0, max_iter=10, tol=1e-6)

    @testset "initialise function tests" begin
        @testset "Basic functionality" begin
            # Get arguments in the format expected by initialise
            args = MLJModelInterface.reformat(eos_model, X_table, y_cat)
            T_instances_test = args[2]
            Tf = args[3]

            weights, distances, loss, inner_fitresult, inner_cache, inner_report = EOS.initialise(
                eos_model, 0, args[1], T_instances_test, Tf
            )

            # Test weights properties
            @test length(weights) == T_instances_test
            @test all(weights .>= 0)
            @test isapprox(sum(weights), 1.0, atol=1e-10)
            @test eltype(weights) == Tf

            # Test distances properties
            @test length(distances) == T_instances_test
            @test all(distances .>= 0)
            @test eltype(distances) == Tf

            # Test loss array properties
            @test length(loss) == eos_model.max_iter + 1
            @test isfinite(loss[1])
            @test all(loss[2:end] .== Tf(Inf))  # Only first element should be set

            # Test that inner model was fitted
            @test inner_fitresult !== nothing
            @test inner_cache !== nothing
            @test inner_report !== nothing
        end

        @testset "Edge cases" begin
            # Test with single instance
            X_single, y_single = make_blobs(
                1, D_features; centers=1, rng=456, as_table=true
            )
            eos_model_single = EOSWrapper(
                eSPAClassifier(K=1); alpha=1.0, max_iter=10, tol=1e-6
            )
            args_single = MLJModelInterface.reformat(eos_model_single, X_single, y_single)
            T_single = args_single[2]
            Tf = args_single[3]

            weights, distances, loss, _, _, _ = EOS.initialise(
                eos_model_single, 0, args_single[1], T_single, Tf
            )

            @test length(weights) == 1
            @test weights[1] ≈ 1.0 atol=1e-10
            @test length(distances) == 1
            @test distances[1] >= 0
        end
    end

    @testset "_fit! function tests" begin
        @testset "Basic iterative optimization" begin
            # Setup initial state
            args = MLJModelInterface.reformat(eos_model, X_table, y_cat)
            T_instances_test = args[2]
            Tf = args[3]

            weights, distances, loss, inner_fitresult, inner_cache, inner_report = EOS.initialise(
                eos_model, 0, args[1], T_instances_test, Tf
            )

            # Create timer output
            to = TimerOutput()

            # Run _fit!
            final_inner_fitresult, final_inner_cache, final_inner_report, iterations, final_to = EOS._fit!(
                weights,
                distances,
                loss,
                inner_fitresult,
                inner_cache,
                inner_report,
                eos_model,
                0,
                args[1],
                to,
            )

            # Test that function completed
            @test iterations >= 1
            @test iterations <= eos_model.max_iter
            @test final_inner_fitresult !== nothing
            @test final_inner_cache !== nothing
            @test final_inner_report !== nothing
            @test final_to isa TimerOutput

            # Test that weights remain valid
            @test all(weights .>= 0)
            @test isapprox(sum(weights), 1.0, atol=1e-10)

            # Test that distances remain valid
            @test all(distances .>= 0)
            @test length(distances) == T_instances_test

            # Test loss array was updated
            @test isfinite(loss[1])
            @test all(isfinite.(loss[1:(iterations + 1)]))
            @test all(loss[(iterations + 2):end] .== Tf(Inf))
        end

        @testset "Loss monotonicity" begin
            # Setup with more iterations to see convergence behavior
            eos_test = EOSWrapper(inner_model; alpha=1.0, max_iter=20, tol=1e-8)
            args = MLJModelInterface.reformat(eos_test, X_table, y_cat)
            T_instances_test = args[2]
            Tf = args[3]

            weights, distances, loss, inner_fitresult, inner_cache, inner_report = EOS.initialise(
                eos_test, 0, args[1], T_instances_test, Tf
            )

            to = TimerOutput()

            _, _, _, iterations, _ = EOS._fit!(
                weights,
                distances,
                loss,
                inner_fitresult,
                inner_cache,
                inner_report,
                eos_test,
                0,
                args[1],
                to,
            )

            # Test that loss generally decreases (allowing for small numerical increases)
            @test loss[iterations + 1] <= loss[1] + 1e-10

            # Test that each step doesn't increase loss significantly
            for i in 2:(iterations + 1)
                @test loss[i] - loss[i - 1] <= 1e-10
            end
        end

        @testset "Uniform weights with alpha=Inf" begin
            # Create EOS model with alpha=Inf - should preserve uniform weights
            eos_uniform = EOSWrapper(inner_model; alpha=Inf, max_iter=10, tol=1e-6)
            args = MLJModelInterface.reformat(eos_uniform, X_table, y_cat)
            T_instances_test = args[2]
            Tf = args[3]

            weights, distances, loss, inner_fitresult, inner_cache, inner_report = EOS.initialise(
                eos_uniform, 0, args[1], T_instances_test, Tf
            )

            to = TimerOutput()

            _, _, _, iterations, _ = EOS._fit!(
                weights,
                distances,
                loss,
                inner_fitresult,
                inner_cache,
                inner_report,
                eos_uniform,
                0,
                args[1],
                to,
            )

            # With alpha=Inf, weights should remain uniform
            expected_weight = Tf(1.0) / T_instances_test
            @test all(isapprox.(weights, expected_weight, atol=1e-10))
        end
    end
end

@testset "2. EOS Update Method Tests" begin
    X, y = MLJBase.@load_iris
    @testset "update equivalence to fit" begin
        # First, train a model to full convergence to get the baseline
        inner_model_full = eSPAClassifier(
            K=3, epsC=1e-3, epsW=1e-1, max_iter=1, tol=1e-8, random_state=42
        )
        eos_model_full = EOSWrapper(inner_model_full; alpha=1.0, max_iter=20, tol=1e-8)
        mach_full = MLJBase.machine(eos_model_full, X, y)
        MLJBase.fit!(mach_full; verbosity=0)

        # Get the total iterations needed for convergence
        total_iterations = MLJBase.report(mach_full).iterations
        @test total_iterations >= 3  # Ensure we have enough iterations to split

        # Choose a split point (about halfway through)
        split_iter = max(2, total_iterations ÷ 2)

        # Train a model with limited iterations
        inner_model_partial = eSPAClassifier(
            K=3, epsC=1e-3, epsW=1e-1, max_iter=1, tol=1e-8, random_state=42
        )
        eos_model_partial = EOSWrapper(
            inner_model_partial; alpha=1.0, max_iter=split_iter, tol=1e-8
        )
        mach_partial = MLJBase.machine(eos_model_partial, X, y)
        MLJBase.fit!(mach_partial; verbosity=0)

        # Verify the partial model didn't fully converge
        partial_iterations = MLJBase.report(mach_partial).iterations
        @test partial_iterations == split_iter

        # Update the partial model to complete training
        remaining_iter = total_iterations - split_iter + 5  # Add buffer for safety
        eos_model_partial.max_iter = remaining_iter
        MLJBase.fit!(mach_partial; verbosity=0)  # This should call update internally

        # Get final results
        fitresult_full = mach_full.fitresult
        fitresult_partial = mach_partial.fitresult
        report_full = MLJBase.report(mach_full)
        report_partial = MLJBase.report(mach_partial)

        # Test that final fitted parameters are equivalent
        weights_full = MLJBase.fitted_params(mach_full).weights
        weights_partial = MLJBase.fitted_params(mach_partial).weights
        @test weights_full ≈ weights_partial atol=1e-10

        # Test that inner model parameters are equivalent
        inner_params_full = MLJBase.fitted_params(mach_full).inner_params
        inner_params_partial = MLJBase.fitted_params(mach_partial).inner_params
        @test inner_params_full.C ≈ inner_params_partial.C atol=1e-10
        @test inner_params_full.W ≈ inner_params_partial.W atol=1e-10
        @test inner_params_full.L ≈ inner_params_partial.L atol=1e-10

        # Test full loss history is equivalent and monotonically decreasing
        @test report_full.loss ≈ report_partial.loss atol=1e-10
        for i in 2:length(report_full.loss)
            @test report_full.loss[i] <= report_full.loss[i - 1] + 1e-10
        end

        # Test that total iterations are equivalent (within small tolerance)
        total_iter_full = report_full.iterations
        total_iter_partial = report_partial.iterations
        @test total_iter_full == total_iter_partial

        # Test that ESS values are equivalent
        @test report_full.ESS ≈ report_partial.ESS atol=1e-10
    end

    @testset "update preserves cache and report structure" begin
        # Test that update properly accumulates timings and loss history

        # Train partially
        inner_model = eSPAClassifier(
            K=3, epsC=1e-3, epsW=1e-1, max_iter=1, tol=1e-8, random_state=101
        )
        eos_model = EOSWrapper(inner_model; alpha=1.0, max_iter=5, tol=1e-8)
        mach = MLJBase.machine(eos_model, X, y)
        MLJBase.fit!(mach; verbosity=0)

        # Get initial report
        report1 = MLJBase.report(mach)
        initial_iterations = report1.iterations
        initial_loss_length = length(report1.loss)

        # Update with more iterations
        eos_model.max_iter = 10
        MLJBase.fit!(mach; verbosity=0)

        # Get updated report
        report2 = MLJBase.report(mach)
        final_iterations = report2.iterations
        final_loss_length = length(report2.loss)

        # Test that iterations and loss history accumulated properly
        @test final_iterations >= initial_iterations
        @test final_loss_length >= initial_loss_length
    end
end
