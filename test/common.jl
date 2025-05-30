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
    end

    @testset "entropy Tests" begin
        # Uniform distribution
        W1 = [0.25, 0.25, 0.25, 0.25]
        @test entropy(W1) ≈ 2*log(2)

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
        @test entropy(W_matrix) ≈ -(0.1*log(0.1) + 0.2*log(0.2) + 0.4*log(0.4) + 0.3*log(0.3))
    end

    @testset "assign_closest Tests" begin
        distances1 = [1.0 5.0 3.0;  # Column 1 min is 0.5 at index 2
                      0.5 2.0 4.0;  # Column 2 min is 1.0 at index 3
                      3.0 1.0 0.5]  # Column 3 min is 0.5 at index 3
        
        @test assign_closest(distances1) == [2, 3, 3]
        @test assign_closest(Float32.(distances1)) == [2, 3, 3]

        distances2 = [1.0 1.0;  # Ties, should pick first
                      1.0 1.0] 
        @test assign_closest(distances2) == [1, 1]

        distances3 = [3.0; 1.0; 2.0] # Single column
        @test assign_closest(distances3) == 2
    end

    @testset "assign_closest! (Dense) Tests" begin
        distances = [1.0 5.0 3.0;
                     0.5 2.0 4.0;
                     3.0 1.0 0.5]
        Gamma = zeros(Float64, 3, 3)
        assign_closest!(Gamma, distances)
        expected_Gamma = [0.0 0.0 0.0;
                          1.0 0.0 0.0;
                          0.0 1.0 1.0]
        @test Gamma == expected_Gamma

        Gamma_int = zeros(Int, 3, 3)
        assign_closest!(Gamma_int, distances)
        @test Gamma_int == expected_Gamma
    end

    @testset "assign_closest! (Sparse) Tests" begin
        distances = [1.0 5.0 3.0 0.2;
                     0.5 2.0 4.0 0.5; 
                     3.0 1.0 0.5 1.0] 
        
        # Create a sparse matrix where each column has one 'true' entry (representing one assignment per item)
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
        @test sum(LS_A, dims=1) ≈ [1.0 1.0]
        @test LS_A[:, 1] ≈ [1.0/4.0, 3.0/4.0]
        @test LS_A[:, 2] ≈ [2.0/6.0, 4.0/6.0]

        B = [1.0 0.0; 0.0 1.0] # Already left stochastic
        @test left_stochastic(B) == B

        C = [0.0 0.0; 0.0 0.0] # Zero matrix
        LS_C = left_stochastic(C)
        @test all(isnan, LS_C) # Division by zero sum
    end

    @testset "left_stochastic! Tests" begin
        A = [1.0 2.0; 3.0 4.0]
        A_copy_for_check = copy(A)
        LS_A_ref = left_stochastic(A_copy_for_check) # Get expected result
        
        left_stochastic!(A) # Modify A in-place
        
        @test sum(A, dims=1) ≈ [1.0 1.0]
        @test A ≈ LS_A_ref

        B = [0.0 0.0; 0.0 0.0]
        left_stochastic!(B)
        @test all(isnan, B)
    end

    @testset "right_stochastic Tests" begin
        A = [1.0 3.0; 2.0 4.0] # Transpose of the left_stochastic test matrix
        A_orig = copy(A)
        RS_A = right_stochastic(A)

        @test A == A_orig # Ensure A is not modified
        @test sum(RS_A, dims=2) ≈ reshape([1.0, 1.0], 2, 1)
        @test RS_A[1, :] ≈ [1.0/4.0, 3.0/4.0]
        @test RS_A[2, :] ≈ [2.0/6.0, 4.0/6.0]

        B = [1.0 0.0; 0.0 1.0] # Already right stochastic
        @test right_stochastic(B) == B

        C = [0.0 0.0; 0.0 0.0] # Zero matrix
        RS_C = right_stochastic(C)
        @test all(isnan, RS_C) # Division by zero sum
    end

    @testset "right_stochastic! Tests" begin
        A = [1.0 3.0; 2.0 4.0]
        A_copy_for_check = copy(A)
        RS_A_ref = right_stochastic(A_copy_for_check) # Get expected result

        right_stochastic!(A) # Modify A in-place

        @test sum(A, dims=2) ≈ reshape([1.0, 1.0], 2, 1)
        @test A ≈ RS_A_ref

        B = [0.0 0.0; 0.0 0.0]
        right_stochastic!(B)
        @test all(isnan, B)
    end

end
