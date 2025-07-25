using Test
using Random
using EntropicLearning
using LinearAlgebra
using SparseArrays
using NearestNeighbors: KDTree, knn, inrange, Chebyshev
using SpecialFunctions: digamma
using Statistics: mean, std
using MLJBase
using StatsBase: sample
using Clustering
using Clustering.Distances: SqEuclidean, WeightedSqEuclidean

# Access eSPA module
import EntropicLearning.eSPA as eSPA

@testset "eSPA extras" begin
    @testset "mi_continuous_discrete function tests" begin

        # Test 1: deterministic threshold, expected MI = ln(2)
        @testset "step threshold MI" begin
            n = 10_000
            rng = Random.MersenneTwister(42)
            x = rand(rng, n)                        # Uniform ramp
            y = (x .>= 0.5)                         # Step at 0.5, convert to Int
            y_int = Int.(y)

            # Reshape x to matrix format (1 feature, n samples)
            X = reshape(x, 1, n)

            mi_est = eSPA.mi_continuous_discrete(
                X, y_int; n_neighbors=3, rng=Random.MersenneTwister(42)
            )

            @test isapprox(mi_est[1], log(2), atol=0.05)
        end

        # Test 2: 4-level uniform quantiser, expected MI = ln(4)
        @testset "uniform four level MI" begin
            n = 10_000
            rng = Random.MersenneTwister(123)
            x = rand(rng, n)
            y = floor.(Int, 4 * x)                  # 4 equal bins: 0,1,2,3

            # Reshape x to matrix format
            X = reshape(x, 1, n)

            mi_est = eSPA.mi_continuous_discrete(
                X, y; n_neighbors=3, rng=Random.MersenneTwister(123)
            )

            @test isapprox(mi_est[1], log(4), atol=0.05)
        end

        # Test 3: independent variables, expected MI = 0
        @testset "independent zero MI" begin
            n = 10_000
            rng = Random.MersenneTwister(2024)
            x = rand(rng, n)
            y = rand(rng, [0, 1], n)                # Independent Bernoulli label

            # Reshape x to matrix format
            X = reshape(x, 1, n)

            mi_est = eSPA.mi_continuous_discrete(
                X, y; n_neighbors=3, rng=Random.MersenneTwister(2024)
            )

            # Independence ⇒ MI ≈ 0 (allow tiny positive bias)
            @test mi_est[1] < 0.02
        end

        # Test 4: constant label, expected MI = 0
        @testset "constant label MI" begin
            n = 5_000
            rng = Random.MersenneTwister(7)
            x = rand(rng, n)
            y = zeros(Int, n)                       # Deterministic target

            # Reshape x to matrix format
            X = reshape(x, 1, n)

            mi_est = eSPA.mi_continuous_discrete(
                X, y; n_neighbors=3, rng=Random.MersenneTwister(7)
            )

            @test mi_est[1] == 0.0
        end
    end

    @testset "compute_mi_cd function tests" begin

        # Test 1: Only one sample (length 1)
        @testset "single sample MI" begin
            y3 = [0]
            c3 = [0.5]
            # n_neighbors would be min(1, 1-1)=0, but internally k is at least 1 if count > 1
            mi3 = eSPA.compute_mi_cd(c3, y3, 1)

            @test mi3 == 0.0 # Should be 0 as only one unique label
        end

        # Test 2: Two samples, one for each label
        @testset "two samples different labels MI" begin
            y4 = [0, 1]
            c4 = [0.1, 1.0]
            mi4 = eSPA.compute_mi_cd(c4, y4, 3)

            # Should be 0 since each label has only 1 sample (count=1, filtered out)
            @test mi4 == 0.0
        end

        # Test 3: Perfect binary separation
        @testset "perfect binary separation MI" begin
            y5 = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
            c5 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            mi5 = eSPA.compute_mi_cd(c5, y5, 3)

            @test mi5 > 0.3 # Should be reasonably high for clear separation
        end

        # Test 4: Three classes test
        @testset "three classes MI" begin
            y6 = [0, 0, 0, 1, 1, 1, 2, 2, 2]
            c6 = [0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            mi6 = eSPA.compute_mi_cd(c6, y6, 2)

            @test mi6 > 0.5 # Should be high for three well-separated classes
        end

        # Test 5: Different n_neighbors parameter
        @testset "different n_neighbors MI" begin
            y7 = [0, 0, 0, 0, 1, 1, 1, 1]
            c7 = [0.1, 0.2, 0.3, 0.4, 0.7, 0.8, 0.9, 1.0]
            mi7_k1 = eSPA.compute_mi_cd(c7, y7, 1)
            mi7_k2 = eSPA.compute_mi_cd(c7, y7, 2)

            @test mi7_k1 > 0.0
            @test mi7_k2 > 0.0
            # Both should be positive for this separated case
        end

        # Test 6: Unbalanced labels
        @testset "unbalanced labels MI" begin
            y8 = [0, 0, 1] # 2 samples for label 0, 1 sample for label 1
            c8 = [0.1, 0.2, 1.0]
            mi8 = eSPA.compute_mi_cd(c8, y8, 2)

            # Label 1 has count=1, gets filtered out, only 2 samples from label 0 remain
            @test mi8 == 0.0
        end
    end

    @testset "mi_continuous_discrete edge case tests" begin

        # Test 1: Feature with zero standard deviation (before noise addition)
        @testset "zero std feature MI" begin
            # Use alternating target
            y_t1 = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]

            X_t4 = ones(Float64, 1, 20) # All values are the same
            y_t4 = y_t1 # Use a varying y
            rng_t4 = Random.MersenneTwister(42)
            mi_t4 = eSPA.mi_continuous_discrete(X_t4, y_t4; n_neighbors=3, rng=rng_t4)

            # After adding noise, the std will not be zero.
            @test mi_t4[1] < 0.01 # Expect low MI
        end

        # Test 2: Multiple features test
        @testset "multiple features MI" begin
            n = 100
            rng = Random.MersenneTwister(42)

            # Create 3 features with different relationships to target
            X_multi = zeros(Float64, 3, n)
            y_multi = rand(rng, [0, 1], n)

            # Feature 1: highly correlated
            X_multi[1, :] = 2.0 * float.(y_multi) + 0.1 * randn(rng, n)
            # Feature 2: weakly correlated
            X_multi[2, :] = 0.3 * float.(y_multi) + 0.5 * randn(rng, n)
            # Feature 3: independent
            X_multi[3, :] = randn(rng, n)

            mi_multi = eSPA.mi_continuous_discrete(
                X_multi, y_multi; n_neighbors=3, rng=Random.MersenneTwister(1)
            )

            # Highly correlated should beat weakly correlated
            @test mi_multi[1] > mi_multi[2]
            @test mi_multi[1] > mi_multi[3] # Highly correlated should beat independent
            @test mi_multi[1] > 0.1 # Highly correlated should have substantial MI
            @test mi_multi[2] < 0.1 # Weakly correlated should have relatively low MI
            @test mi_multi[3] < 0.1 # Independent should have low MI
        end

        # Test 3: Perfect separability
        @testset "perfect separability MI" begin
            X_perfect = zeros(Float64, 1, 10)
            X_perfect[1, 1:5] .= 0.0
            X_perfect[1, 6:10] .= 1.0
            y_perfect = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

            mi_perfect = eSPA.mi_continuous_discrete(
                X_perfect, y_perfect; n_neighbors=3, rng=Random.MersenneTwister(77)
            )

            # Should be high MI, approaching log(2) for binary perfect separation
            @test mi_perfect[1] > 0.5
        end

        # Test 4: Three classes test
        @testset "three classes discrete MI" begin
            n = 60
            X_three = zeros(Float64, 1, n)
            y_three = zeros(Int, n)

            # Create three well-separated groups
            # Class 0 around 0
            X_three[1, 1:20] .= randn(Random.MersenneTwister(1), 20) * 0.1 .+ 0.0
            # Class 1 around 1
            X_three[1, 21:40] .= randn(Random.MersenneTwister(2), 20) * 0.1 .+ 1.0
            # Class 2 around 2
            X_three[1, 41:60] .= randn(Random.MersenneTwister(3), 20) * 0.1 .+ 2.0

            y_three[1:20] .= 0
            y_three[21:40] .= 1
            y_three[41:60] .= 2

            mi_three = eSPA.mi_continuous_discrete(
                X_three, y_three; n_neighbors=3, rng=Random.MersenneTwister(42)
            )

            @test mi_three[1] > 0.8 # Should be high for three well-separated classes
        end

        # Test 5: Different n_neighbors parameter
        @testset "different n_neighbors discrete MI" begin
            n = 50
            rng = Random.MersenneTwister(123)
            x = rand(rng, n)
            y = (x .>= 0.5)
            y_int = Int.(y)
            X = reshape(x, 1, n)

            mi_k1 = eSPA.mi_continuous_discrete(
                X, y_int; n_neighbors=1, rng=Random.MersenneTwister(1)
            )
            mi_k3 = eSPA.mi_continuous_discrete(
                X, y_int; n_neighbors=3, rng=Random.MersenneTwister(1)
            )
            mi_k5 = eSPA.mi_continuous_discrete(
                X, y_int; n_neighbors=5, rng=Random.MersenneTwister(1)
            )

            @test mi_k1[1] > 0.5 # All should be reasonably high for this threshold case
            @test mi_k3[1] > 0.5
            @test mi_k5[1] > 0.5
            # Results may vary slightly with different k values
        end

        # Test 6: Empty matrix edge case
        @testset "empty matrix MI" begin
            X_empty = zeros(Float64, 1, 0) # 1 feature, 0 samples
            y_empty = Int[]

            # This should handle gracefully - though the function may not be designed for
            # this
            # We expect it to either return empty array or handle the edge case
            try
                mi_empty = eSPA.mi_continuous_discrete(
                    X_empty, y_empty; n_neighbors=3, rng=Random.MersenneTwister(42)
                )

                @test length(mi_empty) == 1
            catch e
                println("Test - Empty matrix: Caught expected error: $(typeof(e))")
                @test true # It's acceptable to throw an error for empty input
            end
        end

        # Test 7: Single sample edge case for discrete
        @testset "single sample discrete MI" begin
            X_single = reshape([0.5], 1, 1) # 1 feature, 1 sample
            y_single = [0]

            mi_single = eSPA.mi_continuous_discrete(
                X_single, y_single; n_neighbors=3, rng=Random.MersenneTwister(42)
            )

            @test mi_single[1] == 0.0 # Should be 0 for single sample
        end
    end

    @testset "get_eff function tests" begin
        @testset "basic functionality" begin
            # Test basic properties with normalise=true (default)
            D = 10
            ε = 1.0
            f = eSPA.get_eff(D, ε)

            # Should be between 1/D and 1 when normalised
            @test f >= 1.0 / D
            @test f <= 1.0

            # Test with normalise=false
            f_unnorm = eSPA.get_eff(D, ε; normalise=false)
            @test f_unnorm ≈ f * D atol = 1e-10
            @test f_unnorm >= 1.0
            @test f_unnorm <= D
        end

        @testset "monotonicity properties" begin
            D = 5

            # Should be monotonically increasing with ε
            ε_values = [0.001, 0.01, 0.1, 1.0, 10.0]
            f_values = [eSPA.get_eff(D, ε) for ε in ε_values]

            for i in 2:length(f_values)
                @test f_values[i] >= f_values[i - 1]
            end
        end

        @testset "edge cases" begin
            D = 4

            # Very small ε should give values close to 1/D
            f_small = eSPA.get_eff(D, 1e-10)
            @test f_small ≈ 1.0 / D atol = 0.01

            # Large ε should give values close to 1
            f_large = eSPA.get_eff(D, 1e10)
            @test f_large ≈ 1.0 atol = 0.01

            # Test D=1 case
            f_d1 = eSPA.get_eff(1, 1.0)
            @test f_d1 == 1.0  # Should always be 1 for D=1
        end

        @testset "input validation" begin
            # Test assertions
            @test_throws AssertionError eSPA.get_eff(0, 1.0)    # D must be >= 1
            @test_throws AssertionError eSPA.get_eff(-1, 1.0)   # D must be >= 1
            @test_throws AssertionError eSPA.get_eff(5, 0.0)    # ε must be positive
            @test_throws AssertionError eSPA.get_eff(5, -1.0)   # ε must be positive
        end

        @testset "mathematical properties" begin
            ε = 0.02

            # Test with different D values - larger D should affect scaling
            f_d3 = eSPA.get_eff(3, ε)
            f_d6 = eSPA.get_eff(6, ε)
            # Both should be in valid ranges
            @test f_d3 >= 1.0 / 3 && f_d3 <= 1.0
            @test f_d6 >= 1.0 / 6 && f_d6 <= 1.0
        end
    end

    @testset "get_eps function tests" begin
        @testset "basic functionality" begin
            # Test basic properties with normalise=true (default)
            D = 10
            Deff = 0.5  # Normalised effective dimension
            ε = eSPA.get_eps(D, Deff)

            @test ε > 0.0
            @test isfinite(ε)

            # Test with normalise=false
            Deff_unnorm = 5  # Unnormalised (between 1 and D)
            ε_unnorm = eSPA.get_eps(D, Deff_unnorm; normalise=false)
            @test ε_unnorm > 0.0
            @test isfinite(ε_unnorm)
        end

        @testset "edge cases" begin
            D = 4

            # Minimum effective dimension should return very small ε
            Deff_min = 1.0 / D
            ε_min = eSPA.get_eps(D, Deff_min)
            @test ε_min ≈ eps(Float64) atol = 1e-10

            # Maximum effective dimension should return Inf
            Deff_max = 1.0
            ε_max = eSPA.get_eps(D, Deff_max)
            @test ε_max == Inf

            # Just below minimum (should still return eps)
            Deff_below = 1.0 / D - 1e-10
            ε_below = eSPA.get_eps(D, Deff_below)
            @test ε_below ≈ eps(Float64) atol = 1e-10

            # Just above maximum (should still return Inf)
            Deff_above = 1.0 + 1e-10
            ε_above = eSPA.get_eps(D, Deff_above)
            @test ε_above == Inf

            # Test unnormalised edge cases
            ε_min_unnorm = eSPA.get_eps(D, 1; normalise=false)
            @test ε_min_unnorm ≈ eps(Float64) atol = 1e-10

            ε_max_unnorm = eSPA.get_eps(D, D; normalise=false)
            @test ε_max_unnorm == Inf
        end

        @testset "input validation" begin
            # Test assertions
            @test_throws AssertionError eSPA.get_eps(0, 0.5)     # D must be >= 1
            @test_throws AssertionError eSPA.get_eps(-1, 0.5)    # D must be >= 1
            @test_throws AssertionError eSPA.get_eps(5, 0.0)     # Deff must be positive
            @test_throws AssertionError eSPA.get_eps(5, -0.1)    # Deff must be positive
        end

        @testset "inverse relationship" begin
            # Test that get_eps and get_eff are inverse functions
            D_values = [5, 10, 20]
            ε_values = [0.001, 0.01, 0.1, 1.0, 10.0]

            for D in D_values
                for ε in ε_values
                    # Test normalised case
                    f = eSPA.get_eff(D, ε; normalise=true)
                    ε_recovered = eSPA.get_eps(D, f; normalise=true)
                    @test ε ≈ ε_recovered atol = 1e-10 rtol = 1e-8

                    # Test unnormalised case
                    f_unnorm = eSPA.get_eff(D, ε; normalise=false)
                    ε_recovered_unnorm = eSPA.get_eps(D, f_unnorm; normalise=false)
                    @test ε ≈ ε_recovered_unnorm atol = 1e-10 rtol = 1e-8
                end
            end
        end

        @testset "monotonicity properties" begin
            D = 5

            # Should be monotonically increasing with Deff
            Deff_values = [0.3, 0.5, 0.7, 0.9]
            ε_values = [eSPA.get_eps(D, Deff) for Deff in Deff_values]

            for i in 2:length(ε_values)
                @test ε_values[i - 1] <= ε_values[i]
            end
        end

        @testset "numerical stability" begin
            D = 1000

            # Test values very close to boundaries
            Deff_close_min = 1.0 / D + 1e-15
            Deff_close_max = 1.0 - 1e-15

            ε_close_min = eSPA.get_eps(D, Deff_close_min)
            ε_close_max = eSPA.get_eps(D, Deff_close_max)

            @test isfinite(ε_close_min)
            @test ε_close_min > 0
            @test isfinite(ε_close_max)
            @test ε_close_max > 0
        end
    end
end

@testset "data front-end" begin
    @testset "get_pi function tests" begin
        @testset "basic functionality" begin
            # Test with 3 classes, 5 instances
            y_int = [1, 2, 3, 1, 2]
            M_classes = 3
            Pi_mat = eSPA.get_pi(y_int, M_classes, Float64)

            # Test dimensions
            @test size(Pi_mat) == (3, 5)

            # Test one-hot encoding
            expected = [
                1.0 0.0 0.0 1.0 0.0;
                0.0 1.0 0.0 0.0 1.0;
                0.0 0.0 1.0 0.0 0.0
            ]
            @test Pi_mat ≈ expected

            # Test column sums are 1
            @test all(sum(Pi_mat; dims=1) .≈ 1.0)

            # Test non-negative values
            @test all(Pi_mat .>= 0)
        end

        @testset "edge cases" begin
            # Empty vector
            y_empty = Int[]
            Pi_empty = eSPA.get_pi(y_empty, 3, Float64)
            @test size(Pi_empty) == (3, 0)
            @test eltype(Pi_empty) == Float64

            # Single instance
            y_single = [2]
            Pi_single = eSPA.get_pi(y_single, 4, Float32)
            @test size(Pi_single) == (4, 1)
            @test eltype(Pi_single) == Float32
            @test Pi_single[:, 1] == [0.0f0, 1.0f0, 0.0f0, 0.0f0]

            # All same class
            y_same = [1, 1, 1, 1]
            Pi_same = eSPA.get_pi(y_same, 3, Float64)
            @test size(Pi_same) == (3, 4)
            @test all(Pi_same[1, :] .== 1.0)
            @test all(Pi_same[2:3, :] .== 0.0)
        end

        @testset "different types" begin
            # Test with different integer types
            y_int8 = Int8[1, 2, 1]
            Pi_int8 = eSPA.get_pi(y_int8, 2, Float64)
            @test size(Pi_int8) == (2, 3)
            @test Pi_int8 == [1.0 0.0 1.0; 0.0 1.0 0.0]

            # Test with different float types
            y_int = [1, 2, 3]
            Pi_f32 = eSPA.get_pi(y_int, 3, Float32)
            Pi_f64 = eSPA.get_pi(y_int, 3, Float64)
            @test eltype(Pi_f32) == Float32
            @test eltype(Pi_f64) == Float64
            @test Pi_f32 ≈ Pi_f64
        end
    end

    @testset "format_weights function tests" begin
        @testset "basic functionality" begin
            y = [1, 2, 1, 3]
            w = [0.1, 0.3, 0.2, 0.4]
            formatted = eSPA.format_weights(w, y, Float64)

            # Test normalization
            @test sum(formatted) ≈ 1.0
            @test eltype(formatted) == Float64
            @test length(formatted) == length(y)

            # Test proportional to input
            expected = w ./ sum(w)
            @test formatted ≈ expected
        end

        @testset "type conversion" begin
            y = [1, 2, 3]
            w_int = [1, 2, 3]
            formatted = eSPA.format_weights(w_int, y, Float32)

            @test eltype(formatted) == Float32
            @test formatted ≈ Float32[1 / 6, 2 / 6, 3 / 6]
        end

        @testset "edge cases" begin
            y = [1, 2, 3, 4]

            # All equal weights
            w_equal = [1.0, 1.0, 1.0, 1.0]
            formatted = eSPA.format_weights(w_equal, y)
            @test all(formatted .≈ 0.25)

            # Very small weights
            w_small = [1e-10, 1e-10, 1e-10, 1e-10]
            formatted = eSPA.format_weights(w_small, y)
            @test sum(formatted) ≈ 1.0
            @test all(formatted .≈ 0.25)

            # Single weight
            y_single = [1]
            w_single = [0.5]
            formatted = eSPA.format_weights(w_single, y_single)
            @test formatted ≈ [1.0]
        end

        @testset "error handling" begin
            y = [1, 2, 3]

            # Wrong length
            w_wrong = [0.1, 0.2]
            @test_throws ArgumentError eSPA.format_weights(w_wrong, y)

            # Wrong type
            w_wrong_type = "not_a_vector"
            @test_throws ArgumentError eSPA.format_weights(w_wrong_type, y)

            # Non-numeric weights
            w_non_numeric = ["a", "b", "c"]
            @test_throws ArgumentError eSPA.format_weights(w_non_numeric, y)
        end
    end
end

@testset "eSPA core" begin
    # Test data generation
    D_features = 3
    T_instances = 100
    K_clusters = 3
    # Data in MLJ format
    X_table, y_cat = MLJBase.make_blobs(
        T_instances, D_features; centers=K_clusters, rng=123, as_table=true
    )
    y_int = convert.(Int64, MLJBase.int(y_cat)) # Convert from UInt32 to Int64
    X_transposed = MLJBase.matrix(X_table; transpose=true)
    classes = sort(unique(y_int))
    M_classes = length(classes)

    # Create one-hot encoded targets
    P = zeros(Float64, M_classes, T_instances)
    for t in 1:T_instances
        P[y_int[t], t] = 1.0
    end

    # Create weights
    weights = fill(1.0 / T_instances, T_instances)

    # Create random weights
    rng_weights = Random.MersenneTwister(456)
    weights_random = rand(rng_weights, T_instances)
    weights_random ./= sum(weights_random)

    @testset "1. Initialisation" begin
        # Test with different initialization modes
        for (mi_init, kpp_init) in
            [(true, true), (true, false), (false, true), (false, false)]
            model = eSPAClassifier(;
                K=3, mi_init=mi_init, kpp_init=kpp_init, random_state=42
            )

            C, W, L, G = eSPA.initialise(model, X_transposed, P, y_int)

            # Test dimensions
            @test size(C) == (D_features, 3)
            @test size(W) == (D_features,)
            @test size(L) == (M_classes, 3)
            @test size(G) == (3, T_instances)

            # Test W properties
            @test all(W .>= 0)
            @test sum(W) ≈ 1.0 atol = 1e-10

            # Test L properties (left stochastic)
            @test all(L .>= 0)
            @test all(sum(L; dims=1) .≈ 1.0)

            # Test G properties (sparse assignment matrix)
            @test all(sum(G; dims=1) .== 1)
            @test nnz(G) == T_instances
        end

        # Test edge case: K=1
        model_k1 = eSPAClassifier(; K=1, random_state=42)
        C, W, L, G = eSPA.initialise(model_k1, X_transposed, P, y_int)
        @test size(C) == (D_features, 1)
        @test all(G.rowval .== 1)  # All points assigned to cluster 1

        # Test reproducibility
        model1 = eSPAClassifier(; K=3, random_state=42)
        model2 = eSPAClassifier(; K=3, random_state=42)

        C1, W1, L1, G1 = eSPA.initialise(model1, X_transposed, P, y_int)
        C2, W2, L2, G2 = eSPA.initialise(model2, X_transposed, P, y_int)

        @test C1 ≈ C2
        @test W1 ≈ W2
        @test L1 ≈ L2
        @test G1.rowval == G2.rowval
    end

    @testset "2. Core Update Functions" begin
        # Setup for update function tests
        model = eSPAClassifier(; K=K_clusters, epsC=1e-3, epsW=1e-1, random_state=101)
        C, W, L, G = eSPA.initialise(model, X_transposed, P, y_int)

        @testset "update_G! tests" begin
            G_orig = copy(G)
            loss_before = eSPA.calc_loss(
                X_transposed, P, C, W, L, G, model.epsC, model.epsW, weights
            )

            eSPA.update_G!(G, X_transposed, P, C, W, L, model.epsC, weights)

            # Test that G remains valid assignment matrix
            @test all(sum(G; dims=1) .== 1)
            @test size(G) == (K_clusters, T_instances)
            @test nnz(G) == T_instances

            # Test that loss doesn't increase
            loss_after = eSPA.calc_loss(
                X_transposed, P, C, W, L, G, model.epsC, model.epsW, weights
            )
            @test loss_after <= loss_before + 1e-10

            # Test with epsC = 0.0
            G_zero = copy(G_orig)
            loss_before = eSPA.calc_loss(
                X_transposed, P, C, W, L, G_zero, 0.0, model.epsW, weights
            )

            eSPA.update_G!(G_zero, X_transposed, P, C, W, L, 0.0, weights)

            @test all(sum(G_zero; dims=1) .== 1)
            @test size(G_zero) == (K_clusters, T_instances)
            @test nnz(G_zero) == T_instances

            loss_after = eSPA.calc_loss(
                X_transposed, P, C, W, L, G_zero, 0.0, model.epsW, weights
            )
            @test loss_after <= loss_before + 1e-10
        end

        @testset "update_W! tests" begin
            W_orig = copy(W)
            loss_before = eSPA.calc_loss(
                X_transposed, P, C, W, L, G, model.epsC, model.epsW, weights
            )

            eSPA.update_W!(W, X_transposed, C, G, model.epsW, weights)

            # Test W remains a valid probability vector
            @test all(W .>= 0)
            @test sum(W) ≈ 1.0 atol = 1e-10

            # Test that loss doesn't increase
            loss_after = eSPA.calc_loss(
                X_transposed, P, C, W, L, G, model.epsC, model.epsW, weights
            )
            @test loss_after <= loss_before + 1e-10

            # Test with epsW = Inf
            W_inf = fill(1.0 / D_features, D_features)
            loss_before = eSPA.calc_loss(
                X_transposed, P, C, W_inf, L, G, model.epsC, Inf, weights
            )

            eSPA.update_W!(W_inf, X_transposed, C, G, Inf, weights)

            @test all(W_inf .>= 0)
            @test sum(W_inf) ≈ 1.0 atol = 1e-10
            @test all(W_inf .≈ 1.0 / D_features)

            loss_after = eSPA.calc_loss(
                X_transposed, P, C, W_inf, L, G, model.epsC, Inf, weights
            )
            @test loss_after <= loss_before + 1e-10
        end

        @testset "update_C! tests" begin
            C_orig = copy(C)
            loss_before = eSPA.calc_loss(
                X_transposed, P, C, W, L, G, model.epsC, model.epsW, weights
            )

            eSPA.update_C!(C, X_transposed, G, weights)

            # Test dimensions
            @test size(C) == (D_features, K_clusters)

            # Test that loss doesn't increase
            loss_after = eSPA.calc_loss(
                X_transposed, P, C, W, L, G, model.epsC, model.epsW, weights
            )
            @test loss_after <= loss_before + 1e-10
        end

        @testset "update_L! tests" begin
            L_orig = copy(L)
            loss_before = eSPA.calc_loss(
                X_transposed, P, C, W, L, G, model.epsC, model.epsW, weights
            )

            eSPA.update_L!(L, P, G)

            # Test L properties
            @test all(L .>= 0)
            @test all(sum(L; dims=1) .≈ 1.0)
            @test size(L) == (M_classes, K_clusters)

            # Test that loss doesn't increase
            loss_after = eSPA.calc_loss(
                X_transposed, P, C, W, L, G, model.epsC, model.epsW, weights
            )
            @test loss_after <= loss_before + 1e-10
        end

        @testset "update_P! tests" begin
            P_test = Matrix{Float64}(undef, M_classes, T_instances)
            eSPA.update_P!(P_test, L, G)

            # Test P properties
            @test all(P_test .>= 0)
            @test all(sum(P_test; dims=1) .≈ 1.0)
            @test size(P_test) == (M_classes, T_instances)
        end
    end

    @testset "3. Loss and Convergence Functions" begin
        model = eSPAClassifier(; K=K_clusters, epsC=1e-3, epsW=1e-1, random_state=123)
        C, W, L, G = eSPA.initialise(model, X_transposed, P, y_int)

        @testset "calc_loss tests" begin
            loss = eSPA.calc_loss(
                X_transposed, P, C, W, L, G, model.epsC, model.epsW, weights
            )

            @test isfinite(loss)
            @test isa(loss, Float64)

            # Test with different regularisation parameters
            loss_zero = eSPA.calc_loss(
                X_transposed, P, C, W, L, G, 0.0, model.epsW, weights
            )
            @test isfinite(loss_zero)
            @test isa(loss_zero, Float64)

            loss_inf = eSPA.calc_loss(X_transposed, P, C, W, L, G, model.epsC, Inf, weights)
            @test isfinite(loss_inf)
            @test isa(loss_inf, Float64)
        end

        @testset "Convergence helper functions" begin
            # Test converged function
            loss_seq = [10.0, 5.0, 2.0, 1.9, 1.899999, 1.899998]
            @test !eSPA.converged(loss_seq, 0, 100, 1e-6)  # iter=0 never converged
            @test !eSPA.converged(loss_seq, 3, 100, 1e-6)  # Not converged yet
            @test eSPA.converged(loss_seq, 5, 100, 1e-6)   # Should be converged
            @test eSPA.converged(loss_seq, 5, 5, 1e-6)     # Max iter reached

            # Test check functions (should not throw warnings when verbosity=0)
            loss_seq[2] = 11.0 # Artificially increase loss
            @test_nowarn eSPA.check_loss(loss_seq, 1, 0)
            @test_nowarn eSPA.check_iter(100, 100, 0)

            # Test that warnings are thrown when verbosity > 0
            @test_warn "Loss function has increased at iteration 1 by 1.0" eSPA.check_loss(
                loss_seq, 1, 1
            )
            @test_warn "Maximum number of iterations reached" eSPA.check_iter(100, 100, 1)
            @test_warn "Loss function in test context has increased at iteration 1 by 1.0" eSPA.check_loss(
                loss_seq, 1, 1; context="test context"
            )
            @test_warn "Maximum number of iterations reached in test context" eSPA.check_iter(
                100, 100, 1; context="test context"
            )
        end
    end

    @testset "4. Cluster Management Functions" begin
        # Create test data with some empty clusters
        K_test = 5
        T_test = 20
        G_test = spzeros(Bool, K_test, T_test)
        # Assign points to only clusters 1, 3, 4 (leaving 2 and 5 empty)
        for t in 1:T_test
            cluster = t <= 5 ? 1 : (t <= 10 ? 3 : 4)
            G_test[cluster, t] = true
        end

        @testset "find_empty tests" begin
            notEmpty, K_new = eSPA.find_empty(G_test)
            @test K_new == 3  # Only 3 non-empty clusters
            @test notEmpty == [true, false, true, true, false]

            # Test all empty case
            G_empty = spzeros(Bool, 3, 10)
            notEmpty, K_new = eSPA.find_empty(G_empty)
            @test K_new == 0
            @test all(.!notEmpty)

            # Test none empty case
            G_full = sparse([1, 2, 3], [1, 2, 3], [true, true, true], 3, 3)
            notEmpty, K_new = eSPA.find_empty(G_full)
            @test K_new == 3
            @test all(notEmpty)
        end

        @testset "remove_empty tests" begin
            C_test = randn(D_features, K_test)
            L_test = rand(M_classes, K_test)
            left_stochastic!(L_test)

            notEmpty, K_new = eSPA.find_empty(G_test)
            C_new, L_new, G_new = eSPA.remove_empty(C_test, L_test, G_test, notEmpty)

            # Test dimensions after removal
            @test size(C_new) == (D_features, K_new)
            @test size(L_new) == (M_classes, K_new)
            @test size(G_new) == (K_new, T_test)

            # Test that data is preserved correctly
            @test all(sum(G_new; dims=1) .== 1)
            @test all(sum(L_new; dims=1) .≈ 1.0)
        end
    end

    @testset "5. Prediction Functions" begin
        # Train a model using MLJ interface
        model = eSPAClassifier(; K=K_clusters, epsC=1e-3, epsW=1e-1, random_state=42)
        mach = MLJBase.machine(model, X_table, y_cat)
        MLJBase.fit!(mach; verbosity=0)
        fitresult = mach.fitresult

        @testset "_predict tests" begin
            X_test = X_transposed[:, 1:10]
            P_test, _ = eSPA._predict(model, fitresult.C, fitresult.W, fitresult.L, X_test)
            @test size(P_test) == (M_classes, 10)
            @test all(sum(P_test; dims=1) .≈ 1.0)
            @test all(P_test .>= 0)

            # Test reproducibility
            model_repro = eSPAClassifier(; K=K_clusters, random_state=123)
            P1, _ = eSPA._predict(
                model_repro, fitresult.C, fitresult.W, fitresult.L, X_test
            )
            model_repro2 = eSPAClassifier(; K=K_clusters, random_state=123)
            P2, _ = eSPA._predict(
                model_repro2, fitresult.C, fitresult.W, fitresult.L, X_test
            )
            @test P1 ≈ P2

            # Test single instance prediction
            X_single = X_transposed[:, 1:1]
            P_single, _ = eSPA._predict(
                model, fitresult.C, fitresult.W, fitresult.L, X_single
            )
            @test size(P_single) == (M_classes, 1)
            @test sum(P_single) ≈ 1.0
        end

        @testset "predict tests" begin
            # Test prediction on subset of training data
            X_test = MLJBase.selectrows(X_table, 1:10)
            y_pred = MLJBase.predict(mach, X_test)

            # Test output properties
            @test length(y_pred) == 10
            @test all(MLJBase.classes(y_pred) .== MLJBase.classes(y_cat))

            # Test that each prediction is a valid probability distribution
            for pred in y_pred
                sum_pred = 0.0
                for m in MLJBase.classes(pred)
                    prob = MLJBase.pdf(pred, m)
                    @test prob >= 0.0
                    sum_pred += prob
                end
                @test sum_pred ≈ 1.0 atol = 1e-10
            end

            # Test single instance prediction and empty matrix prediction
            X_single = MLJBase.selectrows(X_table, 1:1)
            X_empty = MLJBase.selectrows(X_table, 1:0)
            y_pred_single = MLJBase.predict(mach, X_single)
            @test length(y_pred_single) == 1
            @test MLJBase.classes(y_pred_single) == MLJBase.classes(y_cat)
            probs = MLJBase.pdf(y_pred_single, MLJBase.classes(y_pred_single))
            @test all(probs .>= 0)
            @test sum(probs) ≈ 1.0 atol = 1e-10
            y_pred_empty = MLJBase.predict(mach, X_empty)
            @test length(y_pred_empty) == 0
            @test MLJBase.classes(y_pred_empty) == MLJBase.classes(y_cat)
        end
    end

    @testset "6. Integration and Property Tests" begin
        # Train a model using MLJ interface
        model = eSPAClassifier(; K=K_clusters, epsC=1e-3, epsW=1e-1, random_state=42)
        mach = MLJBase.machine(model, X_table, y_cat)
        MLJBase.fit!(mach; verbosity=0)
        fitresult = mach.fitresult
        report = MLJBase.report(mach)

        @testset "Monotonic loss decrease" begin
            loss = report.loss
            iterations = report.iterations

            # Test that loss is the correct length
            @test length(loss) == iterations

            # Test overall decrease
            @test loss[end] <= loss[1] + 1e-10

            # Test that loss is monotonically decreasing
            for i in 2:iterations
                @test loss[i] <= loss[i - 1] + 1e-10
            end
        end

        @testset "Matrix property preservation" begin
            G = fitresult.G
            C = fitresult.C
            W = fitresult.W
            L = fitresult.L
            classes = fitresult.classes

            # Test that G is a valid assignment matrix
            @test all(sum(G; dims=1) .== 1)
            @test size(G) == (K_clusters, T_instances)
            @test nnz(G) == T_instances

            # Test that C is a valid centroid matrix
            @test size(C) == (D_features, K_clusters)

            # Test that W is a valid probability vector
            @test all(W .>= 0)
            @test sum(W) ≈ 1.0 atol = 1e-10

            # Test that L is a valid conditional probability matrix
            @test all(L .>= 0)
            @test all(sum(L; dims=1) .≈ 1.0)
            @test size(L) == (M_classes, K_clusters)

            # Test that the classes are correct
            @test classes == MLJBase.classes(y_cat)
        end
    end

    @testset "7. Reproducibility and Robustness" begin
        # Train a model using MLJ interface
        model = eSPAClassifier(; K=K_clusters, epsC=1e-3, epsW=1e-1, random_state=42)
        mach = MLJBase.machine(model, X_table, y_cat)
        MLJBase.fit!(mach; verbosity=0)
        fitresult = mach.fitresult

        @testset "Deterministic behavior" begin
            # Test same RNG seeds produce identical results
            model1 = eSPAClassifier(; K=K_clusters, epsC=1e-3, epsW=1e-1, random_state=42)
            mach1 = MLJBase.machine(model1, X_table, y_cat)
            MLJBase.fit!(mach1; verbosity=0)
            fitresult1 = mach1.fitresult

            @test fitresult.G == fitresult1.G
            @test fitresult.C == fitresult1.C
            @test fitresult.W == fitresult1.W
            @test fitresult.L == fitresult1.L
            @test fitresult.classes == fitresult1.classes
        end

        @testset "Reproducible affiliations" begin
            # Test unbiasing
            _, G_unbias = eSPA._predict(
                model, fitresult.C, fitresult.W, fitresult.L, X_transposed
            )
            @test G_unbias == fitresult.G
        end
    end

    @testset "8. Edge Cases" begin
        # eSPA with epsC = 0 and epsW = Inf should behave like k-means (when mi_init=false)

        # Train a model using MLJ interface
        model = eSPAClassifier(;
            K=K_clusters, epsC=0.0, epsW=Inf, mi_init=false, random_state=Random.Xoshiro(42)
        )
        mach = MLJBase.machine(model, X_table, y_cat)
        MLJBase.fit!(mach; verbosity=0)
        fitresult = mach.fitresult

        # Train a k-means model from Clustering.jl
        rng = Random.Xoshiro(42)
        KMeansResult = kmeans(X_transposed, K_clusters; rng=rng, maxiter=200)

        # First check that W is uniform
        @test all(fitresult.W .≈ 1.0 / D_features)

        # Check that centroids are the same
        @test fitresult.C ≈ KMeansResult.centers
        # Check that assignments are the same
        @test fitresult.G.rowval == KMeansResult.assignments
    end

    @testset "9. MLJ Interface" begin
        # Train a model using MLJ interface
        model = eSPAClassifier(; K=K_clusters, epsC=1e-3, epsW=1e-1, random_state=42)
        mach = MLJBase.machine(model, X_table, y_cat)
        MLJBase.fit!(mach; verbosity=0)
        fitresult = mach.fitresult
        report = MLJBase.report(mach)

        @testset "fitted_params tests" begin
            fitted_params_result = MLJBase.fitted_params(mach)

            # Test that all required fields are present
            @test haskey(fitted_params_result, :C)
            @test haskey(fitted_params_result, :W)
            @test haskey(fitted_params_result, :L)

            # Test that fitted parameters match fitresult
            @test fitted_params_result.C == fitresult.C
            @test fitted_params_result.W == fitresult.W
            @test fitted_params_result.L == fitresult.L
        end

        @testset "feature_importances tests" begin
            feature_importances_result = MLJBase.feature_importances(
                model, fitresult, report
            )

            # Test that result is a vector of pairs
            @test isa(feature_importances_result, Vector)
            @test length(feature_importances_result) == D_features

            # Test that each element is a Pair with Symbol => Float64
            for (i, pair) in enumerate(feature_importances_result)
                @test isa(pair, Pair)
                @test isa(pair.first, Symbol)
                @test isa(pair.second, Float64)
                @test 0.0 <= pair.second <= 1.0
            end
        end
    end

    @testset "10. Core Update Functions - Weighted" begin

        # Setup for weighted update function tests
        model = eSPAClassifier(; K=K_clusters, epsC=1e-3, epsW=1e-1, random_state=101)
        C, W, L, G = eSPA.initialise(model, X_transposed, P, y_int)

        @testset "update_G! with random weights" begin
            G_orig = copy(G)
            loss_before = eSPA.calc_loss(
                X_transposed, P, C, W, L, G, model.epsC, model.epsW, weights_random
            )

            eSPA.update_G!(G, X_transposed, P, C, W, L, model.epsC, weights_random)

            # Test that G remains valid assignment matrix
            @test all(sum(G; dims=1) .== 1)
            @test size(G) == (K_clusters, T_instances)
            @test nnz(G) == T_instances

            # Test that loss doesn't increase
            loss_after = eSPA.calc_loss(
                X_transposed, P, C, W, L, G, model.epsC, model.epsW, weights_random
            )
            @test loss_after <= loss_before + 1e-10

            # Test with epsC = 0.0
            G_zero = copy(G_orig)
            loss_before = eSPA.calc_loss(
                X_transposed, P, C, W, L, G_zero, 0.0, model.epsW, weights_random
            )

            eSPA.update_G!(G_zero, X_transposed, P, C, W, L, 0.0, weights_random)

            @test all(sum(G_zero; dims=1) .== 1)
            @test size(G_zero) == (K_clusters, T_instances)
            @test nnz(G_zero) == T_instances

            loss_after = eSPA.calc_loss(
                X_transposed, P, C, W, L, G_zero, 0.0, model.epsW, weights_random
            )
            @test loss_after <= loss_before + 1e-10
        end

        @testset "update_W! with random weights" begin
            W_orig = copy(W)
            loss_before = eSPA.calc_loss(
                X_transposed, P, C, W, L, G, model.epsC, model.epsW, weights_random
            )

            eSPA.update_W!(W, X_transposed, C, G, model.epsW, weights_random)

            # Test W remains a valid probability vector
            @test all(W .>= 0)
            @test sum(W) ≈ 1.0 atol = 1e-10

            # Test that loss doesn't increase
            loss_after = eSPA.calc_loss(
                X_transposed, P, C, W, L, G, model.epsC, model.epsW, weights_random
            )
            @test loss_after <= loss_before + 1e-10

            # Test with epsW = Inf
            W_inf = fill(1.0 / D_features, D_features)
            loss_before = eSPA.calc_loss(
                X_transposed, P, C, W_inf, L, G, model.epsC, Inf, weights_random
            )

            eSPA.update_W!(W_inf, X_transposed, C, G, Inf, weights_random)

            @test all(W_inf .>= 0)
            @test sum(W_inf) ≈ 1.0 atol = 1e-10
            @test all(W_inf .≈ 1.0 / D_features)

            loss_after = eSPA.calc_loss(
                X_transposed, P, C, W_inf, L, G, model.epsC, Inf, weights_random
            )
            @test loss_after <= loss_before + 1e-10
        end

        @testset "update_C! with random weights" begin
            C_orig = copy(C)
            loss_before = eSPA.calc_loss(
                X_transposed, P, C, W, L, G, model.epsC, model.epsW, weights_random
            )

            eSPA.update_C!(C, X_transposed, G, weights_random)

            # Test dimensions
            @test size(C) == (D_features, K_clusters)

            # Test that loss doesn't increase
            loss_after = eSPA.calc_loss(
                X_transposed, P, C, W, L, G, model.epsC, model.epsW, weights_random
            )
            @test loss_after <= loss_before + 1e-10

            # Test with zero-weight clusters (edge case)
            weights_zero = copy(weights_random)
            weights_zero[G[1, :].nzind] .= 0.0  # Set weights in cluster 1 to zero
            weights_zero ./= sum(weights_zero)  # Renormalize

            C_zero = copy(C_orig)
            @test_nowarn eSPA.update_C!(C_zero, X_transposed, G, weights_zero)
            @test size(C_zero) == (D_features, K_clusters)
            @test all(isfinite.(C_zero))
        end

        @testset "calc_loss with random weights" begin
            loss_random = eSPA.calc_loss(
                X_transposed, P, C, W, L, G, model.epsC, model.epsW, weights_random
            )

            @test isfinite(loss_random)
            @test isa(loss_random, Float64)

            # Test with different regularisation parameters
            loss_zero = eSPA.calc_loss(
                X_transposed, P, C, W, L, G, 0.0, model.epsW, weights_random
            )
            @test isfinite(loss_zero)
            @test isa(loss_zero, Float64)

            loss_inf = eSPA.calc_loss(
                X_transposed, P, C, W, L, G, model.epsC, Inf, weights_random
            )
            @test isfinite(loss_inf)
            @test isa(loss_inf, Float64)
        end
    end

    @testset "11. Integration and Property Tests - Weighted" begin
        # Train a model with weights using MLJ interface
        model = eSPAClassifier(; K=K_clusters, epsC=1e-3, epsW=1e-1, random_state=42)
        mach = MLJBase.machine(model, X_table, y_cat, weights_random)
        MLJBase.fit!(mach; verbosity=0)
        fitresult = mach.fitresult
        report = MLJBase.report(mach)

        @testset "Monotonic loss decrease" begin
            loss = report.loss
            iterations = report.iterations

            # Test that loss is the correct length
            @test length(loss) == iterations

            # Test overall decrease
            @test loss[end] <= loss[1] + 1e-10

            # Test that loss is monotonically decreasing
            for i in 2:iterations
                @test loss[i] <= loss[i - 1] + 1e-10
            end
        end

        @testset "Matrix property preservation" begin
            G = fitresult.G
            C = fitresult.C
            W = fitresult.W
            L = fitresult.L
            classes = fitresult.classes

            # Test that G is a valid assignment matrix
            @test all(sum(G; dims=1) .== 1)
            @test size(G) == (K_clusters, T_instances)
            @test nnz(G) == T_instances

            # Test that C is a valid centroid matrix
            @test size(C) == (D_features, K_clusters)

            # Test that W is a valid probability vector
            @test all(W .>= 0)
            @test sum(W) ≈ 1.0 atol = 1e-10

            # Test that L is a valid conditional probability matrix
            @test all(L .>= 0)
            @test all(sum(L; dims=1) .≈ 1.0)
            @test size(L) == (M_classes, K_clusters)

            # Test that the classes are correct
            @test classes == MLJBase.classes(y_cat)
        end
    end

    @testset "12. Update Method Tests" begin
        @testset "update equivalence to fit" begin
            # Create test data
            X_table, y_cat = MLJBase.make_blobs(150, 4; centers=3, rng=456, as_table=true)

            # First, train a model to full convergence to get the baseline
            model_full = eSPAClassifier(;
                K=3, epsC=1e-3, epsW=1e-1, max_iter=200, tol=1e-8, random_state=42
            )
            mach_full = MLJBase.machine(model_full, X_table, y_cat)
            MLJBase.fit!(mach_full; verbosity=0)

            # Get the total iterations needed for convergence
            total_iterations = MLJBase.report(mach_full).iterations
            @test total_iterations >= 3  # Ensure we have enough iterations to split

            # Choose a split point (about halfway through)
            split_iter = max(2, total_iterations ÷ 2)

            # Train a model with limited iterations
            model_partial = eSPAClassifier(;
                K=3, epsC=1e-3, epsW=1e-1, max_iter=split_iter, tol=1e-8, random_state=42
            )
            mach_partial = MLJBase.machine(model_partial, X_table, y_cat)
            MLJBase.fit!(mach_partial; verbosity=0)

            # Verify the partial model didn't fully converge
            partial_iterations = MLJBase.report(mach_partial).iterations
            @test partial_iterations == split_iter

            # Update the partial model to complete training
            remaining_iter = total_iterations - split_iter + 5  # Add buffer for safety
            model_partial.max_iter = remaining_iter
            MLJBase.fit!(mach_partial; verbosity=0)  # This should call update internally

            # Get final results
            fitresult_full = mach_full.fitresult
            fitresult_partial = mach_partial.fitresult
            report_full = MLJBase.report(mach_full)
            report_partial = MLJBase.report(mach_partial)

            # Test that final fitted parameters are equivalent
            @test fitresult_full.C ≈ fitresult_partial.C atol=1e-10
            @test fitresult_full.W ≈ fitresult_partial.W atol=1e-10
            @test fitresult_full.L ≈ fitresult_partial.L atol=1e-10
            @test fitresult_full.G.rowval == fitresult_partial.G.rowval
            @test fitresult_full.classes == fitresult_partial.classes

            # Test that final loss values are equivalent
            final_loss_full = report_full.loss[end]
            final_loss_partial = report_partial.loss[end]
            @test final_loss_full ≈ final_loss_partial atol=1e-10

            # Test that total iterations are equivalent (within small tolerance)
            total_iter_full = report_full.iterations
            total_iter_partial = report_partial.iterations
            @test abs(total_iter_full - total_iter_partial) <= 1

            # Test that loss sequences match (concatenated partial should equal full)
            # Allow for small differences due to numerical precision
            if total_iter_full == total_iter_partial
                @test report_full.loss ≈ report_partial.loss atol=1e-8
            end
        end

        @testset "update with weights" begin
            # Test update method with sample weights
            X_table, y_cat = MLJBase.make_blobs(100, 3; centers=2, rng=789, as_table=true)
            weights = rand(Random.MersenneTwister(789), 100)
            weights ./= sum(weights)

            # Full training with weights
            model_full = eSPAClassifier(;
                K=2, epsC=1e-3, epsW=1e-1, max_iter=100, tol=1e-8, random_state=123
            )
            mach_full = MLJBase.machine(model_full, X_table, y_cat, weights)
            MLJBase.fit!(mach_full; verbosity=0)

            total_iterations = MLJBase.report(mach_full).iterations
            split_iter = max(2, total_iterations ÷ 2)

            # Partial training with weights
            model_partial = eSPAClassifier(;
                K=2, epsC=1e-3, epsW=1e-1, max_iter=split_iter, tol=1e-8, random_state=123
            )
            mach_partial = MLJBase.machine(model_partial, X_table, y_cat, weights)
            MLJBase.fit!(mach_partial; verbosity=0)

            # Update to completion
            model_partial.max_iter = total_iterations - split_iter + 5
            MLJBase.fit!(mach_partial; verbosity=0)

            # Test equivalence
            fitresult_full = mach_full.fitresult
            fitresult_partial = mach_partial.fitresult

            @test fitresult_full.C ≈ fitresult_partial.C atol=1e-10
            @test fitresult_full.W ≈ fitresult_partial.W atol=1e-10
            @test fitresult_full.L ≈ fitresult_partial.L atol=1e-10
            @test fitresult_full.G.rowval == fitresult_partial.G.rowval
        end

        @testset "update preserves cache and report structure" begin
            # Test that update properly accumulates timings and loss history
            X_table, y_cat = MLJBase.make_blobs(80, 2; centers=2, rng=101, as_table=true)

            # Train partially
            model = eSPAClassifier(;
                K=2, epsC=1e-3, epsW=1e-1, max_iter=5, tol=1e-8, random_state=101
            )
            mach = MLJBase.machine(model, X_table, y_cat)
            MLJBase.fit!(mach; verbosity=0)

            # Get initial report
            report1 = MLJBase.report(mach)
            initial_iterations = report1.iterations
            initial_loss_length = length(report1.loss)

            # Update with more iterations
            model.max_iter = 10
            MLJBase.fit!(mach; verbosity=0)

            # Get updated report
            report2 = MLJBase.report(mach)
            final_iterations = report2.iterations
            final_loss_length = length(report2.loss)

            # Test that iterations and loss history accumulated properly
            @test final_iterations >= initial_iterations
            @test final_loss_length >= initial_loss_length

            # Test that loss is monotonically decreasing
            for i in 2:length(report2.loss)
                @test report2.loss[i] <= report2.loss[i - 1] + 1e-10
            end

            # Test that timings are preserved and updated
            @test haskey(report2.timings, "Training")
            @test haskey(report2.timings, "Initialisation")
        end

        @testset "update with different hyperparameters" begin
            # Test that update works when only changing max_iter
            X_table, y_cat = MLJBase.make_blobs(60, 2; centers=2, rng=202, as_table=true)

            model = eSPAClassifier(;
                K=2, epsC=1e-3, epsW=1e-1, max_iter=3, tol=1e-8, random_state=202
            )
            mach = MLJBase.machine(model, X_table, y_cat)
            MLJBase.fit!(mach; verbosity=0)

            # Store initial state
            fitresult1 = deepcopy(mach.fitresult)
            report1 = MLJBase.report(mach)

            # Update with more iterations
            model.max_iter = 10
            MLJBase.fit!(mach; verbosity=0)

            # Test that the model continued from previous state
            fitresult2 = mach.fitresult
            report2 = MLJBase.report(mach)

            # The final loss should be <= initial loss (improvement or same)
            @test report2.loss[end] <= report1.loss[end] + 1e-10

            # Total iterations should be more than initial
            @test report2.iterations >= report1.iterations

            # Test that feature names and classes are preserved
            @test report1.features == report2.features
            @test report1.classes == report2.classes
        end

        @testset "update convergence behavior" begin
            # Test that update respects convergence tolerance
            X_table, y_cat = MLJBase.make_blobs(50, 2; centers=2, rng=303, as_table=true)

            model = eSPAClassifier(;
                K=2, epsC=1e-3, epsW=1e-1, max_iter=5, tol=1e-6, random_state=303
            )
            mach = MLJBase.machine(model, X_table, y_cat)
            MLJBase.fit!(mach; verbosity=0)

            initial_loss = MLJBase.report(mach).loss[end]

            # If already converged, update should not change much
            model.max_iter = 20
            MLJBase.fit!(mach; verbosity=0)

            final_loss = MLJBase.report(mach).loss[end]

            # Loss should not increase
            @test final_loss <= initial_loss + 1e-10

            # Test that the algorithm respects convergence
            loss_history = MLJBase.report(mach).loss
            if length(loss_history) > 1
                # Check if the last few iterations show convergence
                last_improvement =
                    abs(loss_history[end] - loss_history[end - 1]) /
                    abs(loss_history[end - 1])
                if last_improvement <= model.tol
                    @test true  # Properly converged
                else
                    @test MLJBase.report(mach).iterations == model.max_iter  # Hit max_iter
                end
            end
        end
    end
end
