using Test
using EntropicLearning
using LinearAlgebra # For norm, etc.
using SparseArrays # For sparse matrix tests
using Statistics # For mean, etc.

# Tests for functions in common/functions.jl
@testset "Common Functions Tests" begin
    @testset "safelog Tests" begin
        @test safelog(1.0) == log(1.0)
        @test safelog(smallest) == log(smallest)
        @test safelog(0.0; tol=smallest) == log(smallest)
        @test safelog(-1.0; tol=smallest) == log(smallest)
        @test safelog(exp(2.0)) ≈ 2.0

        v = [1.0, smallest, 0.0, -1.0, exp(2.0)]
        v_safe = [log(1.0), log(smallest), log(smallest), log(smallest), 2.0]
        @test safelog(v; tol=smallest) ≈ v_safe

        @test safelog(0.0; tol=1e-20) == log(1e-20)
        @test safelog([0.0, 1.0]; tol=1e-20) ≈ [log(1e-20), log(1.0)]

        # Test with a matrix
        m = [1.0 exp(3.0); 0.0 -2.0]
        m_safe = [log(1.0) 3.0; log(smallest) log(smallest)]
        @test safelog(m; tol=smallest) ≈ m_safe
    end

    @testset "entropy Tests" begin
        # Uniform distribution
        W1 = [0.25, 0.25, 0.25, 0.25]
        @test entropy(W1) ≈ 2 * log(2)

        # Distribution with zeros
        W2 = [0.5, 0.5, 0.0, 0.0]
        # Expected: -(0.5*log(0.5) + 0.5*log(0.5) + 0*log(smallest) + 0*log(smallest))
        @test entropy(W2; tol=smallest) ≈ -(0.5 * log(0.5) + 0.5 * log(0.5))

        # Single element array
        W3 = [1.0]
        @test entropy(W3) ≈ 0.0

        W4 = [0.0]
        @test entropy(W4; tol=smallest) ≈ 0.0

        # Test with a matrix
        W_matrix = [0.1 0.4; 0.2 0.3]
        @test entropy(W_matrix) ≈
            -(0.1 * log(0.1) + 0.2 * log(0.2) + 0.4 * log(0.4) + 0.3 * log(0.3))
    end

    @testset "cross_entropy Tests" begin
        # Basic matrices (Float64)
        A_m1 = [0.1 0.9; 0.8 0.2]
        B_m1 = [0.2 0.8; 0.7 0.3]
        # Expected: - (0.1*log(0.2) + 0.8*log(0.7) + 0.9*log(0.8) + 0.2*log(0.3))
        expected_ce_m1 = -(
            A_m1[1, 1] * log(B_m1[1, 1]) +
            A_m1[2, 1] * log(B_m1[2, 1]) +
            A_m1[1, 2] * log(B_m1[1, 2]) +
            A_m1[2, 2] * log(B_m1[2, 2])
        )
        @test cross_entropy(A_m1, B_m1) ≈ expected_ce_m1

        # Basic vectors (Float32)
        A_v1 = Float32[0.25, 0.75]
        B_v1 = Float32[0.5, 0.5]
        expected_ce_v1 = -(0.25f0 * log(0.5f0) + 0.75f0 * log(0.5f0))
        @test cross_entropy(A_v1, B_v1) ≈ expected_ce_v1
        # Accumulator promotes
        @test typeof(cross_entropy(A_v1, B_v1)) == promote_type(Float32, Float32, Float64)

        # 3D Arrays
        A_3d = rand(2, 2, 2)
        A_3d ./= sum(A_3d)
        B_3d = rand(2, 2, 2)
        B_3d ./= sum(B_3d) # Make B_3d probabilities too
        expected_ce_3d = 0.0
        for i in eachindex(A_3d, B_3d)
            expected_ce_3d -= A_3d[i] * log(B_3d[i])
        end
        @test cross_entropy(A_3d, B_3d) ≈ expected_ce_3d

        # Test with tol and zeros in B
        A_m2 = [0.5 0.5]
        B_m2_zeros = [0.0 1.0]
        custom_tol = 1e-10
        # Expected: -(0.5*log(custom_tol) + 0.5*log(1.0))
        expected_ce_m2_tol = -(0.5 * log(custom_tol) + 0.5 * log(1.0))
        @test cross_entropy(A_m2, B_m2_zeros; tol=custom_tol) ≈ expected_ce_m2_tol

        # Test with smallest tol (default)
        # Expected: -(0.5*log(smallest) + 0.5*log(1.0))
        expected_ce_m2_smallest =
            -(0.5 * safelog(0.0; tol=EntropicLearning.smallest) + 0.5 * log(1.0))
        @test cross_entropy(A_m2, B_m2_zeros) ≈ expected_ce_m2_smallest

        # DimensionMismatch Tests
        A_m_dim = [1.0 2.0]
        B_m_dim_wrong = [1.0 2.0 3.0]
        @test_throws DimensionMismatch cross_entropy(A_m_dim, B_m_dim_wrong)

        A_v_dim = [1.0]
        B_v_dim_wrong = [1.0, 2.0]
        @test_throws DimensionMismatch cross_entropy(A_v_dim, B_v_dim_wrong)

        # using OffsetArrays
        # A_off = OffsetArray([0.1, 0.9], 0:1)
        # B_off_match = OffsetArray([0.2, 0.8], 0:1)
        # B_off_mismatch = OffsetArray([0.2, 0.8], 1:2)
        # expected_ce_off = -(0.1*log(0.2) + 0.9*log(0.8))
        # @test cross_entropy(A_off, B_off_match) ≈ expected_ce_off
        # @test_throws DimensionMismatch cross_entropy(A_off, B_off_mismatch)

    end

    @testset "assign_closest Tests" begin
        distances1 = [
            1.0 5.0 3.0  # Column 1 min is 0.5 at index 2
            0.5 2.0 4.0  # Column 2 min is 1.0 at index 3
            3.0 1.0 0.5
        ]  # Column 3 min is 0.5 at index 3

        @test assign_closest(distances1) == [2, 3, 3]
        @test assign_closest(Float32.(distances1)) == [2, 3, 3]

        distances2 = [
            1.0 1.0  # Ties, should pick first
            1.0 1.0
        ]
        @test assign_closest(distances2) == [1, 1]

        distances3 = [3.0; 1.0; 2.0] # Single column
        @test assign_closest(distances3) == 2
    end

    @testset "assign_closest! (Dense) Tests" begin
        distances = [
            1.0 5.0 3.0
            0.5 2.0 4.0
            3.0 1.0 0.5
        ]
        Gamma = zeros(Float64, 3, 3)
        assign_closest!(Gamma, distances)
        expected_Gamma = [
            0.0 0.0 0.0
            1.0 0.0 0.0
            0.0 1.0 1.0
        ]
        @test Gamma == expected_Gamma

        Gamma_int = zeros(Int, 3, 3)
        assign_closest!(Gamma_int, distances)
        @test Gamma_int == expected_Gamma
    end

    @testset "assign_closest! (Sparse) Tests" begin
        distances = [
            1.0 5.0 3.0 0.2
            0.5 2.0 4.0 0.5
            3.0 1.0 0.5 1.0
        ]

        # Create a sparse matrix where each column has one 'true' entry (representing one
        # assignment per item)
        K, T = size(distances)
        rowval_initial = rand(1:K, T)
        Gamma_sparse_bool = sparse(rowval_initial, 1:T, ones(Bool, T), K, T)

        assign_closest!(Gamma_sparse_bool, distances)
        expected_rowval = [2, 3, 3, 1] # Correct cluster indices for items 1, 2, 3 & 4
        @test Gamma_sparse_bool.rowval == expected_rowval

        # Test with different initial assignments to ensure they are overwritten
        rowval_initial_2 = [3, 2, 1, 3]
        Gamma_sparse_bool_2 = sparse(rowval_initial_2, 1:T, ones(Bool, T), K, T)
        assign_closest!(Gamma_sparse_bool_2, distances)
        @test Gamma_sparse_bool_2.rowval == expected_rowval
    end

    @testset "left_stochastic Tests" begin
        A = [1.0 2.0; 3.0 4.0]
        A_orig = copy(A)
        LS_A = left_stochastic(A)

        @test A == A_orig # Ensure A is not modified
        @test sum(LS_A; dims=1) ≈ [1.0 1.0]
        @test LS_A[:, 1] ≈ [1.0 / 4.0, 3.0 / 4.0]
        @test LS_A[:, 2] ≈ [2.0 / 6.0, 4.0 / 6.0]

        B = [1.0 0.0; 0.0 1.0] # Already left stochastic
        @test left_stochastic(B) == B

        C = [0.0 0.0; 0.0 0.0] # Zero matrix
        LS_C = left_stochastic(C)
        @test all(LS_C .≈ 0.5) # Fallback to uniform distribution
        @test sum(LS_C; dims=1) ≈ [1.0 1.0] # Each column sums to 1
    end

    @testset "left_stochastic! Tests" begin
        A = [1.0 2.0; 3.0 4.0]
        A_copy_for_check = copy(A)
        LS_A_ref = left_stochastic(A_copy_for_check) # Get expected result

        left_stochastic!(A) # Modify A in-place

        @test sum(A; dims=1) ≈ [1.0 1.0]
        @test A ≈ LS_A_ref

        B = [0.0 0.0; 0.0 0.0]
        left_stochastic!(B)
        @test all(B .≈ 0.5) # Fallback to uniform distribution
        @test sum(B; dims=1) ≈ [1.0 1.0] # Each column sums to 1
    end

    @testset "right_stochastic Tests" begin
        A = [1.0 3.0; 2.0 4.0] # Transpose of the left_stochastic test matrix
        A_orig = copy(A)
        RS_A = right_stochastic(A)

        @test A == A_orig # Ensure A is not modified
        @test sum(RS_A; dims=2) ≈ reshape([1.0, 1.0], 2, 1)
        @test RS_A[1, :] ≈ [1.0 / 4.0, 3.0 / 4.0]
        @test RS_A[2, :] ≈ [2.0 / 6.0, 4.0 / 6.0]

        B = [1.0 0.0; 0.0 1.0] # Already right stochastic
        @test right_stochastic(B) == B

        C = [0.0 0.0; 0.0 0.0] # Zero matrix
        RS_C = right_stochastic(C)
        @test all(RS_C .≈ 0.5) # Fallback to uniform distribution
        @test sum(RS_C; dims=2) ≈ reshape([1.0, 1.0], 2, 1) # Each row sums to 1
    end

    @testset "right_stochastic! Tests" begin
        A = [1.0 3.0; 2.0 4.0]
        A_copy_for_check = copy(A)
        RS_A_ref = right_stochastic(A_copy_for_check) # Get expected result

        right_stochastic!(A) # Modify A in-place

        @test sum(A; dims=2) ≈ reshape([1.0, 1.0], 2, 1)
        @test A ≈ RS_A_ref

        B = [0.0 0.0; 0.0 0.0]
        right_stochastic!(B)
        @test all(B .≈ 0.5) # Fallback to uniform distribution
        @test sum(B; dims=2) ≈ reshape([1.0, 1.0], 2, 1) # Each row sums to 1
    end

    @testset "normalise! Tests" begin
        # Basic normalization test (Float64)
        W = [1.0, 2.0, 3.0]
        W_orig = copy(W)
        normalise!(W)

        @test sum(W) ≈ 1.0
        @test W ≈ W_orig ./ sum(W_orig)  # Should be [1/6, 2/6, 3/6]
        @test W ≈ [1.0 / 6.0, 2.0 / 6.0, 3.0 / 6.0]

        # Test with Float32
        W_f32 = Float32[0.5, 1.5, 2.0]
        W_f32_orig = copy(W_f32)
        normalise!(W_f32)

        @test sum(W_f32) ≈ 1.0f0
        @test W_f32 ≈ W_f32_orig ./ sum(W_f32_orig)

        # Test with already normalized vector
        W_norm = [0.2, 0.3, 0.5]
        W_norm_orig = copy(W_norm)
        normalise!(W_norm)

        @test sum(W_norm) ≈ 1.0
        @test W_norm ≈ W_norm_orig  # Should be unchanged

        # Test with very small sum (should fallback to uniform distribution)
        W_small = [1e-20, 2e-20, 3e-20]  # Sum is much smaller than eps(Float64)
        normalise!(W_small)

        @test sum(W_small) ≈ 1.0
        @test all(W_small .≈ 1.0 / 3.0)  # Should be uniform distribution

        # Test with zero vector
        W_zero = [0.0, 0.0, 0.0, 0.0]
        normalise!(W_zero)

        @test sum(W_zero) ≈ 1.0
        @test all(W_zero .≈ 0.25)  # Should be uniform distribution

        # Test with single element
        W_single = [5.0]
        normalise!(W_single)

        @test sum(W_single) ≈ 1.0
        @test W_single[1] ≈ 1.0

        # Test with single zero element
        W_single_zero = [0.0]
        normalise!(W_single_zero)

        @test sum(W_single_zero) ≈ 1.0
        @test W_single_zero[1] ≈ 1.0

        # Test empty vector (edge case)
        W_empty = Float64[]
        normalise!(W_empty)
        @test length(W_empty) == 0  # Should remain empty

        # Test with very small but non-zero sum at Float32 precision
        W_small_f32 = Float32[1e-10, 2e-10]  # Sum might be smaller than eps(Float32)
        normalise!(W_small_f32)
        @test sum(W_small_f32) ≈ 1.0f0
        # Depending on whether sum is > eps(Float32), could be normalized or uniform
        @test all(W_small_f32 .≥ 0.0f0)  # At least check non-negative
    end

    @testset "Softmax Functions Tests" begin
        # Helper function for checking sum-to-one property
        # dim=1 for columns, dim=2 for rows (not used here, but general)
        function check_sum_to_one(M, dim)
            if ndims(M) == 1
                return sum(M) ≈ 1.0
            else
                return all(sum(M; dims=dim) .≈ 1.0)
            end
        end

        @testset "softmax! (Matrix, G provided)" begin
            # Basic Float64
            A_f64 = [1.0 0.0; 0.0 1.0]
            G_f64 = similar(A_f64)
            A_orig_f64 = copy(A_f64)
            @test softmax!(G_f64, A_f64) === nothing
            # A is scaled by prefactor=1.0, so effectively unchanged if prefactor not given
            @test A_f64 == A_orig_f64
            @test G_f64[:, 1] ≈
                [exp(1.0) / (exp(1.0) + exp(0.0)), exp(0.0) / (exp(1.0) + exp(0.0))]
            @test G_f64[:, 2] ≈
                [exp(0.0) / (exp(0.0) + exp(1.0)), exp(1.0) / (exp(0.0) + exp(1.0))]
            @test check_sum_to_one(G_f64, 1)

            # Basic Float32 with prefactor
            A_f32 = Float32[1.0 2.0; 3.0 4.0]
            G_f32 = similar(A_f32)
            A_orig_f32 = copy(A_f32)
            pf = 0.5f0
            softmax!(G_f32, A_f32; prefactor=pf)
            A_scaled_f32 = A_orig_f32 ./ pf
            @test A_f32 ≈ A_scaled_f32 # A is mutated by prefactor
            col1_exp_scaled = exp.(A_scaled_f32[:, 1] .- maximum(A_scaled_f32[:, 1]))
            col2_exp_scaled = exp.(A_scaled_f32[:, 2] .- maximum(A_scaled_f32[:, 2]))
            @test G_f32[:, 1] ≈ col1_exp_scaled ./ sum(col1_exp_scaled)
            @test G_f32[:, 2] ≈ col2_exp_scaled ./ sum(col2_exp_scaled)
            @test check_sum_to_one(G_f32, 1)

            # Empty input
            A_empty = Matrix{Float64}(undef, 0, 2)
            G_empty = similar(A_empty)
            @test softmax!(G_empty, A_empty) === nothing
            @test size(G_empty) == (0, 2)

            A_empty2 = Matrix{Float64}(undef, 2, 0)
            G_empty2 = similar(A_empty2)
            @test softmax!(G_empty2, A_empty2) === nothing # loop over columns won't run
            @test size(G_empty2) == (2, 0)

            # All -Inf
            A_neg_inf = fill(-Inf, 2, 2)
            G_neg_inf = similar(A_neg_inf)
            A_orig_neg_inf = copy(A_neg_inf)
            softmax!(G_neg_inf, A_neg_inf)
            @test A_neg_inf == A_orig_neg_inf # -Inf / 1.0 is still -Inf
            @test all(G_neg_inf .≈ 0.5)
            @test check_sum_to_one(G_neg_inf, 1)

            # Very small numbers (potential underflow to zero sum)
            A_small = [1e-300 1e-300; 1e-300 1e-300]
            G_small = similar(A_small)
            softmax!(G_small, A_small; prefactor=1e-200) # Makes numbers extremely small
            @test all(G_small .≈ 0.5) # Should fallback to uniform
            @test check_sum_to_one(G_small, 1)

            # Prefactor error
            A_err = [1.0 0.0; 0.0 1.0]
            G_err = similar(A_err)
            @test_throws ArgumentError softmax!(G_err, A_err; prefactor=0.0)
            @test_throws ArgumentError softmax!(G_err, A_err; prefactor=-1.0)
        end

        @testset "softmax! (Vector, W provided)" begin
            b_f64 = [1.0, 0.0, 1.0]
            w_f64 = similar(b_f64)
            b_orig_f64 = copy(b_f64)
            @test softmax!(w_f64, b_f64) === nothing
            @test b_f64 == b_orig_f64 # b is scaled by prefactor=1.0
            norm_factor = exp(1.0) + exp(0.0) + exp(1.0)
            @test w_f64 ≈
                [exp(1.0) / norm_factor, exp(0.0) / norm_factor, exp(1.0) / norm_factor]
            @test check_sum_to_one(w_f64, 1)

            b_f32 = Float32[1.0, 2.0, 3.0]
            w_f32 = similar(b_f32)
            b_orig_f32 = copy(b_f32)
            pf = 0.1f0
            softmax!(w_f32, b_f32; prefactor=pf)
            b_scaled_f32 = b_orig_f32 ./ pf
            @test b_f32 ≈ b_scaled_f32 # b is mutated
            b_exp_scaled = exp.(b_scaled_f32 .- maximum(b_scaled_f32))
            @test w_f32 ≈ b_exp_scaled ./ sum(b_exp_scaled)
            @test check_sum_to_one(w_f32, 1)

            # Empty input
            b_empty = Float64[]
            w_empty = similar(b_empty)
            @test softmax!(w_empty, b_empty) === nothing
            @test length(w_empty) == 0

            # All -Inf
            b_neg_inf = fill(-Inf, 3)
            w_neg_inf = similar(b_neg_inf)
            b_orig_neg_inf = copy(b_neg_inf)
            softmax!(w_neg_inf, b_neg_inf)
            @test b_neg_inf == b_orig_neg_inf
            @test all(w_neg_inf .≈ 1 / 3)
            @test check_sum_to_one(w_neg_inf, 1)

            # Prefactor error
            b_err = [1.0, 0.0]
            w_err = similar(b_err)
            @test_throws ArgumentError softmax!(w_err, b_err; prefactor=0.0)
        end
    end
end
