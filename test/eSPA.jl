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
using EntropicLearning.eSPA: eSPAFitResult

# Include the core and extras module functions
include("../src/eSPA/core.jl")
include("../src/eSPA/extras.jl")


@testset "extras" begin
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

            mi_est = mi_continuous_discrete(
                X, y_int; n_neighbors=3, rng=Random.MersenneTwister(42)
            )

            # println("Test 1 - Step threshold:")
            # println("Estimated MI: $(mi_est[1])")
            # println("Expected MI: $(log(2))")

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

            mi_est = mi_continuous_discrete(
                X, y; n_neighbors=3, rng=Random.MersenneTwister(123)
            )

            # println("Test 2 - Four level quantizer:")
            # println("Estimated MI: $(mi_est[1])")
            # println("Expected MI: $(log(4))")

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

            mi_est = mi_continuous_discrete(
                X, y; n_neighbors=3, rng=Random.MersenneTwister(2024)
            )

            # println("Test 3 - Independent variables:")
            # println("Estimated MI: $(mi_est[1])")
            # println("Expected MI: ≈ 0")

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

            mi_est = mi_continuous_discrete(
                X, y; n_neighbors=3, rng=Random.MersenneTwister(7)
            )

            # println("Test 4 - Constant label:")
            # println("Estimated MI: $(mi_est[1])")
            # println("Expected MI: 0.0")

            @test mi_est[1] == 0.0
        end
    end

    @testset "compute_mi_cd function tests" begin

        # Test 1: Only one sample (length 1)
        @testset "single sample MI" begin
            y3 = [0]
            c3 = [0.5]
            # n_neighbors would be min(1, 1-1)=0, but internally k is at least 1 if count > 1
            mi3 = compute_mi_cd(c3, y3, 1)

            # println("Test - Single sample:")
            # println("Estimated MI: $(mi3)")
            # println("Expected MI: 0.0 (only one unique label)")

            @test mi3 == 0.0 # Should be 0 as only one unique label
        end

        # Test 2: Two samples, one for each label
        @testset "two samples different labels MI" begin
            y4 = [0, 1]
            c4 = [0.1, 1.0]
            mi4 = compute_mi_cd(c4, y4, 3)

            # println("Test - Two samples, different labels:")
            # println("Estimated MI: $(mi4)")
            # println("Expected MI: 0.0 (each label has count=1, filtered out)")

            # Should be 0 since each label has only 1 sample (count=1, filtered out)
            @test mi4 == 0.0
        end

        # Test 3: Perfect binary separation
        @testset "perfect binary separation MI" begin
            y5 = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
            c5 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            mi5 = compute_mi_cd(c5, y5, 3)

            # println("Test - Perfect binary separation:")
            # println("Estimated MI: $(mi5)")
            # println("Expected MI: > 0.3 (high MI for clear separation)")

            @test mi5 > 0.3 # Should be reasonably high for clear separation
        end

        # Test 4: Three classes test
        @testset "three classes MI" begin
            y6 = [0, 0, 0, 1, 1, 1, 2, 2, 2]
            c6 = [0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            mi6 = compute_mi_cd(c6, y6, 2)

            # println("Test - Three classes:")
            # println("Estimated MI: $(mi6)")
            # println("Expected MI: > 0.5 (high MI for three separated classes)")

            @test mi6 > 0.5 # Should be high for three well-separated classes
        end

        # Test 5: Different n_neighbors parameter
        @testset "different n_neighbors MI" begin
            y7 = [0, 0, 0, 0, 1, 1, 1, 1]
            c7 = [0.1, 0.2, 0.3, 0.4, 0.7, 0.8, 0.9, 1.0]
            mi7_k1 = compute_mi_cd(c7, y7, 1)
            mi7_k2 = compute_mi_cd(c7, y7, 2)

            # println("Test - Different n_neighbors:")
            # println("MI with k=1: $(mi7_k1)")
            # println("MI with k=2: $(mi7_k2)")
            # println("Expected: Both > 0, may differ slightly")

            @test mi7_k1 > 0.0
            @test mi7_k2 > 0.0
            # Both should be positive for this separated case
        end

        # Test 6: Unbalanced labels
        @testset "unbalanced labels MI" begin
            y8 = [0, 0, 1] # 2 samples for label 0, 1 sample for label 1
            c8 = [0.1, 0.2, 1.0]
            mi8 = compute_mi_cd(c8, y8, 2)

            # println("Test - Unbalanced labels:")
            # println("Estimated MI: $(mi8)")
            # println("Expected MI: 0.0 (label 1 has only 1 sample, gets filtered)")

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
            mi_t4 = mi_continuous_discrete(X_t4, y_t4; n_neighbors=3, rng=rng_t4)

            # println("Test - Zero std feature:")
            # println("Estimated MI: $(mi_t4[1])")
            # println("Expected MI: < 0.01 (low MI due to noise)")

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

            mi_multi = mi_continuous_discrete(
                X_multi, y_multi; n_neighbors=3, rng=Random.MersenneTwister(1)
            )

            # println("Test - Multiple features:")
            # println("MI feature 1 (high corr): $(mi_multi[1])")
            # println("MI feature 2 (weak corr): $(mi_multi[2])")
            # println("MI feature 3 (independent): $(mi_multi[3])")

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

            mi_perfect = mi_continuous_discrete(
                X_perfect, y_perfect; n_neighbors=3, rng=Random.MersenneTwister(77)
            )

            # println("Test - Perfect separability:")
            # println("Estimated MI: $(mi_perfect[1])")
            # println("Expected MI: > 0.5 (close to log(2) ≈ 0.693)")

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

            mi_three = mi_continuous_discrete(
                X_three, y_three; n_neighbors=3, rng=Random.MersenneTwister(42)
            )

            # println("Test - Three classes:")
            # println("Estimated MI: $(mi_three[1])")
            # println("Expected MI: > 0.8 (high MI for three separated classes)")

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

            mi_k1 = mi_continuous_discrete(
                X, y_int; n_neighbors=1, rng=Random.MersenneTwister(1)
            )
            mi_k3 = mi_continuous_discrete(
                X, y_int; n_neighbors=3, rng=Random.MersenneTwister(1)
            )
            mi_k5 = mi_continuous_discrete(
                X, y_int; n_neighbors=5, rng=Random.MersenneTwister(1)
            )

            # println("Test - Different n_neighbors (discrete):")
            # println("MI with k=1: $(mi_k1[1])")
            # println("MI with k=3: $(mi_k3[1])")
            # println("MI with k=5: $(mi_k5[1])")

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
                mi_empty = mi_continuous_discrete(
                    X_empty, y_empty; n_neighbors=3, rng=Random.MersenneTwister(42)
                )
                # println("Test - Empty matrix:")
                # println("MI result: $(mi_empty)")
                # Should return array of length 1 (for 1 feature)
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

            mi_single = mi_continuous_discrete(
                X_single, y_single; n_neighbors=3, rng=Random.MersenneTwister(42)
            )

            # println("Test - Single sample discrete:")
            # println("Estimated MI: $(mi_single[1])")
            # println("Expected MI: 0.0 (only one unique label)")

            @test mi_single[1] == 0.0 # Should be 0 for single sample
        end
    end
end

@testset "core" begin
    # Test data generation
    K_clusters = 3
    X, y = MLJBase.make_blobs(100, 3; centers=K_clusters, rng=123, as_table=false)
    X_transposed = X'  # Transpose for core functions which expect (D, T) format

    # Basic setup
    D_features, T_instances = size(X_transposed)
    classes = unique(y)
    M_classes = length(classes)
    y_int = [findfirst(==(yi), classes) for yi in y]

    # Create one-hot encoded targets
    P = zeros(Float64, M_classes, T_instances)
    for t in 1:T_instances
        P[y_int[t], t] = 1.0
    end

    @testset "1. Initialisation" begin
        # Test with different initialization modes
        for (mi_init, kpp_init) in [(true, true), (true, false), (false, true), (false, false)]
            model = eSPAClassifier(K=3, mi_init=mi_init, kpp_init=kpp_init, random_state=42)

            C, W, L, G = eSPA.initialise(
                model, X_transposed, y_int, D_features, T_instances, M_classes
            )

            # Test dimensions
            @test size(C) == (D_features, 3)
            @test size(W) == (D_features,)
            @test size(L) == (M_classes, 3)
            @test size(G) == (3, T_instances)

            # Test W properties
            @test all(W .>= 0)
            @test sum(W) ≈ 1.0 atol=1e-10

            # Test L properties (left stochastic)
            @test all(L .>= 0)
            @test all(sum(L; dims=1) .≈ 1.0)

            # Test G properties (sparse assignment matrix)
            @test all(sum(G; dims=1) .== 1)
            @test nnz(G) == T_instances
        end

        # Test edge case: K=1
        model_k1 = eSPAClassifier(K=1, random_state=42)
        C, W, L, G = eSPA.initialise(
            model_k1, X_transposed, y_int, D_features, T_instances, M_classes
        )
        @test size(C) == (D_features, 1)
        @test all(G.rowval .== 1)  # All points assigned to cluster 1

        # Test reproducibility
        model1 = eSPAClassifier(K=3, random_state=42)
        model2 = eSPAClassifier(K=3, random_state=42)

        C1, W1, L1, G1 = eSPA.initialise(model1, X_transposed, y_int, D_features, T_instances, M_classes)
        C2, W2, L2, G2 = eSPA.initialise(model2, X_transposed, y_int, D_features, T_instances, M_classes)

        @test C1 ≈ C2
        @test W1 ≈ W2
        @test L1 ≈ L2
        @test G1.rowval == G2.rowval

    end

    @testset "2. Core Update Functions" begin
        # Setup for update function tests
        model = eSPAClassifier(K=K_clusters, epsC=1e-3, epsW=1e-1, random_state=101)
        C, W, L, G = eSPA.initialise(model, X_transposed, y_int, D_features, T_instances, M_classes)

        @testset "update_G! tests" begin
            G_orig = copy(G)
            loss_before = eSPA.calc_loss(X_transposed, P, C, W, L, G, model.epsC, model.epsW)

            eSPA.update_G!(G, X_transposed, P, C, W, L, model.epsC)

            # Test that G remains valid assignment matrix
            @test all(sum(G; dims=1) .== 1)
            @test size(G) == (K_clusters, T_instances)
            @test nnz(G) == T_instances

            # Test that loss doesn't increase
            loss_after = eSPA.calc_loss(X_transposed, P, C, W, L, G, model.epsC, model.epsW)
            @test loss_after <= loss_before + 1e-10

            # Test with epsC = 0.0
            G_zero = copy(G_orig)
            loss_before = eSPA.calc_loss(X_transposed, P, C, W, L, G_zero, 0.0, model.epsW)

            eSPA.update_G!(G_zero, X_transposed, P, C, W, L, 0.0)

            @test all(sum(G_zero; dims=1) .== 1)
            @test size(G_zero) == (K_clusters, T_instances)
            @test nnz(G_zero) == T_instances

            loss_after = eSPA.calc_loss(X_transposed, P, C, W, L, G_zero, 0.0, model.epsW)
            @test loss_after <= loss_before + 1e-10
        end

        @testset "update_W! tests" begin
            W_orig = copy(W)
            loss_before = eSPA.calc_loss(X_transposed, P, C, W, L, G, model.epsC, model.epsW)

            eSPA.update_W!(W, X_transposed, C, G, model.epsW)

            # Test W remains a valid probability vector
            @test all(W .>= 0)
            @test sum(W) ≈ 1.0 atol=1e-10

            # Test that loss doesn't increase
            loss_after = eSPA.calc_loss(X_transposed, P, C, W, L, G, model.epsC, model.epsW)
            @test loss_after <= loss_before + 1e-10

            # Test with epsW = Inf
            W_inf = fill(1.0 / D_features, D_features)
            loss_before = eSPA.calc_loss(X_transposed, P, C, W_inf, L, G, model.epsC, Inf)

            eSPA.update_W!(W_inf, X_transposed, C, G, Inf)

            @test all(W_inf .>= 0)
            @test sum(W_inf) ≈ 1.0 atol=1e-10
            @test all(W_inf .≈ 1.0/D_features)

            loss_after = eSPA.calc_loss(X_transposed, P, C, W_inf, L, G, model.epsC, Inf)
            @test loss_after <= loss_before + 1e-10


        end

        @testset "update_C! tests" begin
            C_orig = copy(C)
            loss_before = eSPA.calc_loss(X_transposed, P, C, W, L, G, model.epsC, model.epsW)

            eSPA.update_C!(C, X_transposed, G)

            # Test dimensions
            @test size(C) == (D_features, K_clusters)

            # Test that loss doesn't increase
            loss_after = eSPA.calc_loss(X_transposed, P, C, W, L, G, model.epsC, model.epsW)
            @test loss_after <= loss_before + 1e-10
        end

        @testset "update_L! tests" begin
            L_orig = copy(L)
            loss_before = eSPA.calc_loss(X_transposed, P, C, W, L, G, model.epsC, model.epsW)

            eSPA.update_L!(L, P, G)

            # Test L properties
            @test all(L .>= 0)
            @test all(sum(L; dims=1) .≈ 1.0)
            @test size(L) == (M_classes, K_clusters)

            # Test that loss doesn't increase
            loss_after = eSPA.calc_loss(X_transposed, P, C, W, L, G, model.epsC, model.epsW)
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
        model = eSPAClassifier(K=K_clusters, epsC=1e-3, epsW=1e-1, random_state=123)
        C, W, L, G = eSPA.initialise(model, X_transposed, y_int, D_features, T_instances, M_classes)

        @testset "calc_loss tests" begin
            loss = eSPA.calc_loss(X_transposed, P, C, W, L, G, model.epsC, model.epsW)

            @test isfinite(loss)
            @test isa(loss, Float64)

            # Test with different regularisation parameters
            loss_zero = eSPA.calc_loss(X_transposed, P, C, W, L, G, 0.0, model.epsW)
            @test isfinite(loss_zero)
            @test isa(loss_zero, Float64)

            loss_inf = eSPA.calc_loss(X_transposed, P, C, W, L, G, model.epsC, Inf)
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
        # Create training data using MLJ format
        X_table, y_int = MLJBase.make_blobs(100, 3; centers=K_clusters, rng=42, as_table=true)
        y_categorical = MLJBase.categorical(y_int)

        # Train a model using MLJ interface
        model = eSPAClassifier(K=K_clusters, epsC=1e-3, epsW=1e-1, random_state=42)
        mach = MLJBase.machine(model, X_table, y_categorical)
        MLJBase.fit!(mach, verbosity=0)
        fitresult = mach.fitresult
        X_transposed = MLJBase.matrix(X_table; transpose=true)

        @testset "predict_proba tests" begin
            X_test = X_transposed[:, 1:10]
            # Test with iterative_pred = false
            P = eSPA.predict_proba(model, fitresult, X_test)
            @test size(P) == (M_classes, 10)
            @test all(sum(P; dims=1) .≈ 1.0)
            @test all(P .>= 0)

            # Test with iterative_pred = true
            model_iter_pred = eSPAClassifier(
                K=K_clusters, iterative_pred=true, random_state=42
            )
            P_iter = eSPA.predict_proba(model_iter_pred, fitresult, X_test)
            @test size(P_iter) == (M_classes, 10)
            @test all(sum(P_iter; dims=1) .≈ 1.0)
            @test all(P_iter .>= 0)

            # Test reproducibility
            for iterative_pred in [true, false]
                model_repro = eSPAClassifier(K=K_clusters, iterative_pred=iterative_pred, random_state=123)
                P1 = eSPA.predict_proba(model_repro, fitresult, X_test)

                model_repro2 = eSPAClassifier(K=K_clusters, iterative_pred=iterative_pred, random_state=123)
                P2 = eSPA.predict_proba(model_repro2, fitresult, X_test)
                @test P1 ≈ P2
            end

            # Test single instance prediction
            X_single = X_transposed[:, 1:1]
            P_single = eSPA.predict_proba(model, fitresult, X_single)
            @test size(P_single) == (M_classes, 1)
            @test sum(P_single) ≈ 1.0
        end

        @testset "predict tests" begin
            # Test prediction on subset of training data
            X_test = MLJBase.selectrows(X_table, 1:10)
            y_pred = MLJBase.predict(mach, X_test)

            # Test output properties
            @test length(y_pred) == 10
            # Test that each prediction is a valid probability distribution
            for pred in y_pred
                sum_pred = 0.0
                for m in MLJBase.classes(pred)
                    prob = MLJBase.pdf(pred, m)
                    @test prob >= 0.0
                    sum_pred += prob
                end
                @test sum_pred ≈ 1.0 atol=1e-10
            end

            # Test with iterative prediction
            model_iter = eSPAClassifier(K=K_clusters, iterative_pred=true, random_state=42)
            mach_iter = MLJBase.machine(model_iter, X_table, y_categorical)
            MLJBase.fit!(mach_iter, verbosity=0)

            y_pred_iter = MLJBase.predict(mach_iter, X_test)
            @test length(y_pred_iter) == 10
            # Test that each prediction is a valid probability distribution
            for pred in y_pred_iter
                sum_pred = 0.0
                for m in MLJBase.classes(pred)
                    prob = MLJBase.pdf(pred, m)
                    @test prob >= 0.0
                    sum_pred += prob
                end
                @test sum_pred ≈ 1.0 atol=1e-10
            end
        end
    end

    @testset "6. Integration and Property Tests" begin
        # Create training data using MLJ format
        T_instances = 100
        D_features = 3
        X_table, y_int = MLJBase.make_blobs(T_instances, D_features; centers=K_clusters, rng=42, as_table=true)
        y_categorical = MLJBase.categorical(y_int)

        # Train a model using MLJ interface
        model = eSPAClassifier(K=K_clusters, epsC=1e-3, epsW=1e-1, random_state=42)
        mach = MLJBase.machine(model, X_table, y_categorical)
        MLJBase.fit!(mach, verbosity=0)
        fitresult = mach.fitresult
        report = MLJBase.report(mach)

        @testset "Monotonic loss decrease" begin
            loss = report.loss
            iterations = report.iterations

            # Test that loss is the correct length
            @test length(loss) == iterations + 1

            # Test overall decrease
            @test loss[end] <= loss[1] + 1e-10

            # Test that loss is monotonically decreasing
            for i in 1:iterations
                @test loss[i+1] <= loss[i] + 1e-10
            end
        end

        @testset "Matrix property preservation" begin
            G = report.G
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
            @test sum(W) ≈ 1.0 atol=1e-10

            # Test that L is a valid conditional probability matrix
            @test all(L .>= 0)
            @test all(sum(L; dims=1) .≈ 1.0)
            @test size(L) == (M_classes, K_clusters)

            # Test that the classes are correct
            @test classes == MLJBase.classes(y_categorical)
        end

    end

    # @testset "7. Reproducibility and Robustness" begin
    #     @testset "Deterministic behavior" begin
    #         # Test same RNG seeds produce identical results
    #         model = eSPAClassifier(K=3, epsC=1e-3, epsW=1e-1, random_state=42)

    #         rng1 = Random.MersenneTwister(123)
    #         C1, W1, L1, G1 = eSPA.initialise(model, X_transposed, y_int, D_features, T_instances, M_classes; rng=rng1)

    #         rng2 = Random.MersenneTwister(123)
    #         C2, W2, L2, G2 = eSPA.initialise(model, X_transposed, y_int, D_features, T_instances, M_classes; rng=rng2)

    #         @test C1 ≈ C2
    #         @test W1 ≈ W2
    #         @test L1 ≈ L2
    #         @test G1.rowval == G2.rowval

    #         # Test updates are deterministic
    #         eSPA.update_G!(G1, X_transposed, P, C1, W1, L1, model.epsC)
    #         eSPA.update_G!(G2, X_transposed, P, C2, W2, L2, model.epsC)
    #         @test G1.rowval == G2.rowval
    #     end

    #     @testset "Reproducible affiliations" begin
    #         # Create training data using MLJ format
    #         X_train, y_train = MLJBase.make_blobs(50, 3; centers=2, rng=42, as_table=false)
    #         X_table = MLJBase.table(X_train)
    #         y_categorical = MLJBase.categorical(y_train)

    #         # Test unbias=true case
    #         model_unbias = eSPAClassifier(K=3, epsC=1e-3, epsW=1e-1, max_iter=5, unbias=true, random_state=42)
    #         mach_unbias = MLJBase.machine(model_unbias, X_table, y_categorical)
    #         MLJBase.fit!(mach_unbias, verbosity=0)

    #         # Test prediction on training data
    #         y_pred_unbias = MLJBase.predict(mach_unbias, X_table)
    #         @test length(y_pred_unbias) == length(y_categorical)
    #         # Test that each prediction is a valid probability distribution
    #         for pred in y_pred_unbias
    #             @test sum(MLJBase.pdf(pred, MLJBase.classes(pred))) ≈ 1.0
    #         end

    #         # Test unbias=true + iterative_pred=true case
    #         model_iter = eSPAClassifier(K=3, epsC=1e-3, epsW=1e-1, max_iter=5, unbias=true,
    #                                    iterative_pred=true, random_state=42)
    #         mach_iter = MLJBase.machine(model_iter, X_table, y_categorical)
    #         MLJBase.fit!(mach_iter, verbosity=0)

    #         # Test prediction on training data
    #         y_pred_iter = MLJBase.predict(mach_iter, X_table)
    #         @test length(y_pred_iter) == length(y_categorical)
    #         # Test that each prediction is a valid probability distribution
    #         for pred in y_pred_iter
    #             @test sum(MLJBase.pdf(pred, MLJBase.classes(pred))) ≈ 1.0
    #         end

    #         # Test reproducibility: same model parameters should give same results
    #         model_repro = eSPAClassifier(K=3, epsC=1e-3, epsW=1e-1, max_iter=5, unbias=true, random_state=42)
    #         mach_repro = MLJBase.machine(model_repro, X_table, y_categorical)
    #         MLJBase.fit!(mach_repro, verbosity=0)

    #         y_pred_repro = MLJBase.predict(mach_repro, X_table)

    #         # Check that predictions are reproducible (same random_state should give same results)
    #         @test length(y_pred_repro) == length(y_pred_unbias)

    #         # Extract fitted parameters to verify internal consistency
    #         fitted_params_unbias = MLJBase.fitted_params(mach_unbias)
    #         fitted_params_repro = MLJBase.fitted_params(mach_repro)

    #         @test fitted_params_unbias.C ≈ fitted_params_repro.C
    #         @test fitted_params_unbias.W ≈ fitted_params_repro.W
    #         @test fitted_params_unbias.L ≈ fitted_params_repro.L
    #     end
    # end

    @testset "8. Edge Cases" begin
        # eSPA with epsC = 0 and epsW = Inf should behave like k-means (when mi_init=false)
        T_instances = 100
        D_features = 3
        X_table, y_int = MLJBase.make_blobs(T_instances, D_features; centers=K_clusters, rng=42, as_table=true)
        y_categorical = MLJBase.categorical(y_int)
        X_transposed = MLJBase.matrix(X_table; transpose=true)

        # Train a model using MLJ interface
        model = eSPAClassifier(K=K_clusters, epsC=0.0, epsW=Inf, mi_init=false, random_state=42)
        mach = MLJBase.machine(model, X_table, y_categorical)
        MLJBase.fit!(mach, verbosity=0)
        fitresult = mach.fitresult
        report = MLJBase.report(mach)

        # Train a k-means model from Clustering.jl
        rng = Random.MersenneTwister(42)
        KMeansResult = kmeans(X_transposed, K_clusters; rng=rng, maxiter=200)

        # First check that W is uniform
        @test all(fitresult.W .≈ 1.0/D_features)

        # Check that centroids are the same
        @test fitresult.C ≈ KMeansResult.centers
        # Check that assignments are the same
        @test report.G.rowval == KMeansResult.assignments
    end
end
