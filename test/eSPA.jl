using Test
using Random
using EntropicLearning
using LinearAlgebra
using NearestNeighbors: KDTree, knn, inrange, Chebyshev
using SpecialFunctions: digamma
using Statistics: mean, std

# Include the extras module functions
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
