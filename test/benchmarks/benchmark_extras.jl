#!/usr/bin/env julia

# Benchmark script for src/eSPA/extras.jl functions
using TimerOutputs
using Random
using LinearAlgebra
using NearestNeighbors: KDTree, knn, inrange, Chebyshev
using SpecialFunctions: digamma
using Statistics: mean, std

# Include the extras functions
include("src/eSPA/extras.jl")

println("=== Performance Benchmark for eSPA extras.jl ===")
println()

# Function to time repeated measurements
function time_repeat(name, func, n=10)
    times = Float64[]
    for _ in 1:n
        t = @elapsed func()
        push!(times, t)
    end
    median_time = sort(times)[div(length(times)+1, 2)]
    println("  $(name): $(round(median_time * 1000, digits=2)) ms (median of $(n) runs)")
    return median_time
end

# Benchmark compute_mi_cd function
function benchmark_compute_mi_cd()
    println("📊 Benchmarking compute_mi_cd function:")
    
    # Small dataset
    c_small = rand(100)
    d_small = rand([0,1], 100)
    time_repeat("Small (100)", () -> compute_mi_cd(c_small, d_small, 3))
    
    # Medium dataset  
    c_medium = rand(1000)
    d_medium = rand([0,1,2], 1000)
    time_repeat("Medium (1000)", () -> compute_mi_cd(c_medium, d_medium, 3))
    
    # Large dataset
    c_large = rand(5000)
    d_large = rand([0,1,2,3], 5000)
    time_repeat("Large (5000)", () -> compute_mi_cd(c_large, d_large, 3), 3)
    
    println()
end

# Benchmark mi_continuous_discrete function (single feature)
function benchmark_mi_single()
    println("📊 Benchmarking mi_continuous_discrete (single feature):")
    
    rng = Random.MersenneTwister(42)
    
    # Small
    x_small = rand(100)
    y_small = rand([0,1], 100)
    time_repeat("Small (100)", () -> mi_continuous_discrete(x_small, y_small; n_neighbors=3, rng=rng))
    
    # Medium
    x_medium = rand(1000)
    y_medium = rand([0,1,2], 1000)
    time_repeat("Medium (1000)", () -> mi_continuous_discrete(x_medium, y_medium; n_neighbors=3, rng=rng))
    
    # Large
    x_large = rand(5000) 
    y_large = rand([0,1,2,3], 5000)
    time_repeat("Large (5000)", () -> mi_continuous_discrete(x_large, y_large; n_neighbors=3, rng=rng), 3)
    
    println()
end

# Benchmark mi_continuous_discrete function (multiple features)
function benchmark_mi_multiple()
    println("📊 Benchmarking mi_continuous_discrete (multiple features):")
    
    rng = Random.MersenneTwister(42)
    
    # Small dataset (5 features, 100 samples)
    X_small = rand(Float64, 5, 100)
    y_small = rand([0, 1, 2], 100)
    time_repeat("Small (5×100)", () -> mi_continuous_discrete(X_small, y_small; n_neighbors=3, rng=rng))
    
    # Medium dataset (20 features, 1000 samples)
    X_medium = rand(Float64, 20, 1000)
    y_medium = rand([0, 1, 2, 3], 1000)
    time_repeat("Medium (20×1000)", () -> mi_continuous_discrete(X_medium, y_medium; n_neighbors=3, rng=rng))
    
    # Large dataset (50 features, 5000 samples)
    X_large = rand(Float64, 50, 5000)
    y_large = rand([0, 1, 2, 3, 4], 5000)
    time_repeat("Large (50×5000)", () -> mi_continuous_discrete(X_large, y_large; n_neighbors=3, rng=rng), 3)
    
    println()
end

# Memory allocation profiling
function profile_allocations()
    println("📊 Memory allocation analysis:")
    
    # Medium size for analysis
    X = rand(Float64, 20, 1000)
    y = rand([0, 1, 2, 3], 1000)
    x_single = X[1, :]
    rng = Random.MersenneTwister(42)
    
    # Profile compute_mi_cd
    result = @timed compute_mi_cd(x_single, y, 3)
    println("  compute_mi_cd: $(round(result.time * 1000, digits=2)) ms, $(round(result.bytes / 1024^2, digits=2)) MB allocated")
    
    # Profile mi_continuous_discrete (single)
    result = @timed mi_continuous_discrete(x_single, y; n_neighbors=3, rng=rng)
    println("  mi_continuous_discrete (single): $(round(result.time * 1000, digits=2)) ms, $(round(result.bytes / 1024^2, digits=2)) MB allocated")
    
    # Profile mi_continuous_discrete (multiple)
    result = @timed mi_continuous_discrete(X, y; n_neighbors=3, rng=rng)
    println("  mi_continuous_discrete (multiple): $(round(result.time * 1000, digits=2)) ms, $(round(result.bytes / 1024^2, digits=2)) MB allocated")
    
    println()
end

# Run all benchmarks
function run_benchmarks()
    benchmark_compute_mi_cd()
    benchmark_mi_single()
    benchmark_mi_multiple()
    profile_allocations()
    
    println("✅ Benchmark complete! Check results above for performance bottlenecks.")
end

# Execute if run directly
if abspath(PROGRAM_FILE) == @__FILE__
    run_benchmarks()
end