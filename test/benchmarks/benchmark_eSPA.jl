#!/usr/bin/env julia

"""
Benchmark script for eSPA core functions.
This script systematically tests the scaling behavior of eSPA core functions
with respect to both the number of features (D) and instances (T), verifying
expected linear scaling relationships.
"""

using EntropicLearning
using StatsBase: sample, median
using LinearAlgebra
using Random
using SparseArrays
using Clustering: initseeds!, KmppAlg, copyseeds!
using Clustering.Distances: SqEuclidean, WeightedSqEuclidean
using TimerOutputs
using NearestNeighbors: KDTree, knn, inrange, Chebyshev
using SpecialFunctions: digamma
using Statistics: mean, std
using Distributions: MultivariateNormal
using MLJBase
using Printf

# Access eSPA module
import EntropicLearning.eSPA as eSPA
using EntropicLearning.eSPA: eSPAFitResult

# Include the core and extras module functions
include("../../src/eSPA/core.jl")
include("../../src/eSPA/extras.jl")
include("../../src/common/functions.jl")

# Function to create synthetic worms data for benchmarking
function make_worms(D::Ti, T::Ti; σ::Tf=1.0, μ::Tf=2.0, random_state::Union{AbstractRNG,Integer} = Random.default_rng()) where {Ti <: Integer, Tf <: AbstractFloat}
    # Get the random number generator
    rng = get_rng(random_state)

    # Initialise X and Π
    X = rand(rng, Tf, D, T)
    P = zeros(2, T)

    # Generate the data - First two dimensions come from 3 MvGaussians
    part = Int(floor(T / 3))    # Instances per cluster
    rem = mod(T, part)

    # First cluster
    X[1:2, 1:part] = rand(rng, MultivariateNormal(µ .* [1, -1], σ .* [1 1; 1 2]), part)
    P[1, 1:part] = ones(1, part)

    # Second cluster
    X[1:2, (part + 1):(2 * part)] = rand(
        rng, MultivariateNormal(µ .* [0, 0], σ .* [1 1; 1 2]), part
    )
    P[1, (part + 1):(2 * part)] = zeros(1, part)

    # Third cluster
    X[1:2, (2 * part + 1):end] = rand(
        rng, MultivariateNormal(µ .* [-1, 1], σ .* [1 1; 1 2]), part + rem
    )
    P[1, (2 * part + 1):end] = ones(1, part + rem)

    # Get probabilities of second class
    P[2, :] = 1 .- P[1, :]

    # Scale the data
    X = X .- minimum(X, dims=2)
    X = X ./ maximum(abs.(X), dims=2)

    return X, P
end

# Enhanced benchmark result structure
struct BenchmarkResult
    func_name::String
    D::Int
    T::Int
    times::Vector{Float64}      # All N run times
    time_median::Float64        # Median time over N runs
    time_std::Float64           # Standard deviation
    memory_allocated::Int       # Bytes allocated per run (median)
end

# Function to create data in MLJ format
function create_test_data(D_features::Int, T_instances::Int; rng::Union{AbstractRNG,Integer} = Random.default_rng())
    # Get worms data
    X, P = make_worms(D_features, T_instances; random_state=rng)
    y = [argmax(view(P, :, t)) for t in axes(P, 2)]

    # Convert to MLJ format
    X_table = MLJBase.table(X')
    y_cat = MLJBase.categorical(y)

    return X_table, y_cat
end

# Scaling test parameter definitions
const T_SCALING_PARAMS = [
    (D=10, T=100),
    (D=10, T=1_000),
    (D=10, T=10_000),
    (D=10, T=100_000),
    (D=10, T=1_000_000)
]

const D_SCALING_PARAMS = [
    (D=10, T=100),
    (D=100, T=100),
    (D=1_000, T=100),
    (D=10_000, T=100),
    (D=100_000, T=100)
]

# Function to create data and initialise parameters for direct testing
function create_test(D_features::Int, T_instances::Int; rng::Union{AbstractRNG,Integer} = Random.default_rng())
    # Get worms data
    X, P = make_worms(D_features, T_instances; random_state=rng)
    y = [argmax(view(P, :, t)) for t in axes(P, 2)]
    M_classes = size(P, 1)
    K_clusters = 3

    # Initialise the model
    model = eSPAClassifier(K=K_clusters, epsC=1e-3, epsW=1e-1, unbias=true, iterative_pred=true, random_state=rng)

    # Initialise parameters
    C, W, L, G = initialise(model, X, y, D_features, T_instances, M_classes)

    return X, P, C, W, L, G, model
end

# Enhanced benchmark function with memory measurement
function benchmark_function_with_memory(func_name::String, func, args...; N_runs::Int=10)
    # Warmup
    func(args...)

    # Collect all timing and memory data
    times = Vector{Float64}(undef, N_runs)
    memories = Vector{Int}(undef, N_runs)

    for i in 1:N_runs
        GC.gc()  # Consistent memory state
        memories[i] = @allocated times[i] = @elapsed func(args...)
    end

    # Extract D and T from the test setup (assumes create_test was used)
    D = size(args[2], 1)  # X matrix is second argument
    T = size(args[2], 2)  # X matrix is second argument

    return BenchmarkResult(
        func_name, D, T, times,
        median(times), std(times),
        Int(median(memories))
    )
end

# Linear scaling analysis
function analyze_scaling(results::Vector{BenchmarkResult}, param::Symbol)
    # Extract parameter values and median times
    if param == :D
        x_vals = Float64[r.D for r in results]
    else  # param == :T
        x_vals = Float64[r.T for r in results]
    end
    y_vals = Float64[r.time_median for r in results]

    # Fit linear model: y = α + β×x
    n = length(x_vals)
    x_mean = mean(x_vals)
    y_mean = mean(y_vals)

    β = sum((x_vals .- x_mean) .* (y_vals .- y_mean)) / sum((x_vals .- x_mean).^2)
    α = y_mean - β * x_mean

    # Calculate R²
    y_pred = α .+ β .* x_vals
    ss_res = sum((y_vals .- y_pred).^2)
    ss_tot = sum((y_vals .- y_mean).^2)
    r_squared = 1 - ss_res / ss_tot

    return (slope=β, intercept=α, r_squared=r_squared)
end

# Enhanced benchmark functions for individual core functions
function benchmark_update_G!(D::Int, T::Int, N_runs::Int=10)
    rng = MersenneTwister(42)
    X, P, C, W, L, G, model = create_test(D, T; rng=rng)

    return benchmark_function_with_memory(
        "update_G!",
        (G, X, P, C, W, L, epsC) -> update_G!(G, X, P, C, W, L, epsC),
        G, X, P, C, W, L, model.epsC;
        N_runs=N_runs
    )
end

function benchmark_update_W!(D::Int, T::Int, N_runs::Int=10)
    rng = MersenneTwister(42)
    X, P, C, W, L, G, model = create_test(D, T; rng=rng)

    return benchmark_function_with_memory(
        "update_W!",
        (W, X, C, G, epsW) -> update_W!(W, X, C, G, epsW),
        W, X, C, G, model.epsW;
        N_runs=N_runs
    )
end

function benchmark_update_C!(D::Int, T::Int, N_runs::Int=10)
    rng = MersenneTwister(42)
    X, P, C, W, L, G, model = create_test(D, T; rng=rng)

    return benchmark_function_with_memory(
        "update_C!",
        (C, X, G) -> update_C!(C, X, G),
        C, X, G;
        N_runs=N_runs
    )
end

function benchmark_update_L!(D::Int, T::Int, N_runs::Int=10)
    rng = MersenneTwister(42)
    X, P, C, W, L, G, model = create_test(D, T; rng=rng)

    return benchmark_function_with_memory(
        "update_L!",
        (L, P, G) -> update_L!(L, P, G),
        L, P, G;
        N_runs=N_runs
    )
end

function benchmark_calc_loss(D::Int, T::Int, N_runs::Int=10)
    rng = MersenneTwister(42)
    X, P, C, W, L, G, model = create_test(D, T; rng=rng)

    return benchmark_function_with_memory(
        "calc_loss",
        (X, P, C, W, L, G, epsC, epsW) -> calc_loss(X, P, C, W, L, G, epsC, epsW),
        X, P, C, W, L, G, model.epsC, model.epsW;
        N_runs=N_runs
    )
end

# Benchmark individual function across parameter range
function benchmark_function_scaling(benchmark_func, params, func_name::String; N_runs::Int=10)
    println("Benchmarking $func_name...")
    results = BenchmarkResult[]

    for (i, param) in enumerate(params)
        print("  Testing D=$(param.D), T=$(param.T) ($i/$(length(params)))... ")
        result = benchmark_func(param.D, param.T, N_runs)
        push!(results, result)
        println("$(round(result.time_median * 1000, digits=6)) ms")
    end

    return results
end

# Reporting function
function print_scaling_results(func_name::String, t_results::Vector{BenchmarkResult}, d_results::Vector{BenchmarkResult})
    println("\n$func_name Scaling Analysis:")

    # T-scaling analysis
    t_analysis = analyze_scaling(t_results, :T)
    println("  T-scaling (Fixed D=10): slope=$(Printf.@sprintf("%.2e", t_analysis.slope)) s/T, R²=$(round(t_analysis.r_squared, digits=3))")

    # D-scaling analysis
    d_analysis = analyze_scaling(d_results, :D)
    println("  D-scaling (Fixed T=100): slope=$(Printf.@sprintf("%.2e", d_analysis.slope)) s/D, R²=$(round(d_analysis.r_squared, digits=3))")

    return t_analysis, d_analysis
end

# Main scaling benchmark function
function run_scaling_benchmarks(; N_runs::Int=10)
    println("=== eSPA Core Functions Scaling Analysis ===")
    println("Testing scaling behavior with D ∈ [10, 100K] and T ∈ [100, 1M]")
    println("Number of runs per test: $N_runs\n")

    functions_to_test = [
        ("update_G!", benchmark_update_G!),
        ("update_W!", benchmark_update_W!),
        ("update_C!", benchmark_update_C!),
        ("update_L!", benchmark_update_L!),
        ("calc_loss", benchmark_calc_loss)
    ]

    all_results = Dict{String, Tuple{Vector{BenchmarkResult}, Vector{BenchmarkResult}}}()
    all_analyses = Dict{String, Tuple{NamedTuple, NamedTuple}}()

    for (func_name, benchmark_func) in functions_to_test
        println("\n" * "="^60)

        # T-scaling benchmarks (fixed D=10)
        println("\nT-Scaling Tests (Fixed D=10):")
        t_results = benchmark_function_scaling(benchmark_func, T_SCALING_PARAMS, func_name; N_runs=N_runs)

        # D-scaling benchmarks (fixed T=100)
        println("\nD-Scaling Tests (Fixed T=100):")
        d_results = benchmark_function_scaling(benchmark_func, D_SCALING_PARAMS, func_name; N_runs=N_runs)

        # Store results
        all_results[func_name] = (t_results, d_results)

        # Analyze and print scaling
        t_analysis, d_analysis = print_scaling_results(func_name, t_results, d_results)
        all_analyses[func_name] = (t_analysis, d_analysis)
    end

    # Summary table
    println("\n" * "="^80)
    println("SCALING SUMMARY")
    println("="^80)
    println(Printf.@sprintf("%-12s | %-12s | %-5s | %-12s | %-5s | %s", "Function", "T-Slope (s/T)", "T-R²", "D-Slope (s/D)", "D-R²", "Status"))
    println("-"^80)

    for func_name in ["update_G!", "update_W!", "update_C!", "update_L!", "calc_loss"]
        t_analysis, d_analysis = all_analyses[func_name]

        # Determine status
        t_status = t_analysis.r_squared > 0.95 ? "✓" : "⚠"
        d_status = d_analysis.r_squared > 0.95 ? "✓" : "⚠"

        # Special case for update_L! - should be constant in D
        if func_name == "update_L!" && abs(d_analysis.slope) < 1e-9
            d_status = "✓ (const)"
        end

        overall_status = (t_status == "✓" && (d_status == "✓" || d_status == "✓ (const)")) ? "Linear" : "Check"

        println(Printf.@sprintf("%-12s | %-12s | %-5.3f | %-12s | %-5.3f | %s",
            func_name,
            Printf.@sprintf("%.2e", t_analysis.slope),
            t_analysis.r_squared,
            Printf.@sprintf("%.2e", d_analysis.slope),
            d_analysis.r_squared,
            overall_status))
    end

    return all_results, all_analyses
end

# Run the benchmark if called directly
if abspath(PROGRAM_FILE) == @__FILE__
    results, analyses = run_scaling_benchmarks(N_runs=10)
    println("\nScaling benchmark complete!")
end
