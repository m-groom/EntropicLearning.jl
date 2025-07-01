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
using Dates
using JSON3

# Access eSPA module
import EntropicLearning.eSPA as eSPA
using EntropicLearning.eSPA: eSPAFitResult

# Include the core and extras module functions
include("../../src/eSPA/core.jl")
include("../../src/eSPA/extras.jl")
include("../../src/common/functions.jl")

# Function to create synthetic worms data for benchmarking
function make_worms(
    D::Ti,
    T::Ti;
    σ::Tf=1.0,
    μ::Tf=2.0,
    random_state::Union{AbstractRNG,Integer}=Random.default_rng(),
) where {Ti<:Integer,Tf<:AbstractFloat}
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
    P[2, :] = 1 .- view(P, 1, :)

    # Scale the data
    X .-= minimum(X; dims=2)
    X ./= maximum(abs.(X); dims=2)

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
function create_test_data(
    D_features::Int, T_instances::Int; rng::Union{AbstractRNG,Integer}=Random.default_rng()
)
    # Get worms data
    X, P = make_worms(D_features, T_instances; random_state=rng)
    y = [argmax(view(P, :, t)) for t in axes(P, 2)]

    # Convert to MLJ format
    X_table = MLJBase.table(X')
    y_cat = MLJBase.categorical(y)

    return X_table, y_cat
end

# Scaling test parameter definitions - doubling progression
const T_SCALING_PARAMS = [
    (D=10, T=100),
    (D=10, T=200),
    (D=10, T=400),
    (D=10, T=800),
    (D=10, T=1600),
    (D=10, T=3200),
    (D=10, T=6400),
    (D=10, T=12800),
    (D=10, T=25600),
    (D=10, T=51200),
    (D=10, T=102400),
    (D=10, T=204800),
    (D=10, T=409600),
    (D=10, T=819200),
]

const D_SCALING_PARAMS = [
    (D=10, T=100),
    (D=20, T=100),
    (D=40, T=100),
    (D=80, T=100),
    (D=160, T=100),
    (D=320, T=100),
    (D=640, T=100),
    (D=1280, T=100),
    (D=2560, T=100),
    (D=5120, T=100),
    (D=10240, T=100),
    (D=20480, T=100),
    (D=40960, T=100),
    (D=81920, T=100),
]

# Function to create data and initialise parameters for direct testing
function create_test(
    D_features::Int, T_instances::Int; rng::Union{AbstractRNG,Integer}=Random.default_rng()
)
    # Get worms data
    X, P = make_worms(D_features, T_instances; random_state=rng)
    y = [argmax(view(P, :, t)) for t in axes(P, 2)]
    M_classes = size(P, 1)
    K_clusters = 3

    # Initialise the model
    model = eSPAClassifier(;
        K=K_clusters,
        epsC=1e-3,
        epsW=1e-1,
        unbias=true,
        iterative_pred=true,
        random_state=rng,
    )

    # Initialise parameters
    C, W, L, G = initialise(model, X, y, D_features, T_instances, M_classes)

    return X, P, C, W, L, G, model
end

# Enhanced benchmark function with memory measurement
function benchmark_function_with_memory(
    func_name::String, func, D::Int, T::Int, args...; N_runs::Int=10
)
    # Warmup
    func(args...)

    # Collect all timing and memory data
    times = Vector{Float64}(undef, N_runs)
    memories = Vector{Int}(undef, N_runs)

    for i in 1:N_runs
        GC.gc()  # Consistent memory state
        memories[i] = @allocated times[i] = @elapsed func(args...)
    end

    return BenchmarkResult(
        func_name, D, T, times, median(times), std(times), Int(median(memories))
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

    # Check for constant y values (avoid NaN)
    if std(y_vals) < 1e-15
        # Essentially constant - return special values
        return (slope=0.0, intercept=y_vals[1], r_squared=1.0)
    end

    # Fit linear model: y = α + β×x
    n = length(x_vals)
    x_mean = mean(x_vals)
    y_mean = mean(y_vals)

    denom = sum((x_vals .- x_mean) .^ 2)
    if denom < 1e-15
        # Constant x values - should not happen with our test setup
        return (slope=0.0, intercept=y_mean, r_squared=0.0)
    end

    β = sum((x_vals .- x_mean) .* (y_vals .- y_mean)) / denom
    α = y_mean - β * x_mean

    # Calculate R²
    y_pred = α .+ β .* x_vals
    ss_res = sum((y_vals .- y_pred) .^ 2)
    ss_tot = sum((y_vals .- y_mean) .^ 2)

    r_squared = if ss_tot < 1e-15
        1.0  # Perfect fit for constant data
    else
        1 - ss_res / ss_tot
    end

    return (slope=β, intercept=α, r_squared=r_squared)
end

# Enhanced benchmark functions for individual core functions
function benchmark_update_G!(D::Int, T::Int, N_runs::Int=10)
    rng = MersenneTwister(42)
    X, P, C, W, L, G, model = create_test(D, T; rng=rng)

    return benchmark_function_with_memory(
        "update_G!",
        (G, X, P, C, W, L, epsC) -> update_G!(G, X, P, C, W, L, epsC),
        D,
        T,
        G,
        X,
        P,
        C,
        W,
        L,
        model.epsC;
        N_runs=N_runs,
    )
end

function benchmark_update_W!(D::Int, T::Int, N_runs::Int=10)
    rng = MersenneTwister(42)
    X, P, C, W, L, G, model = create_test(D, T; rng=rng)

    return benchmark_function_with_memory(
        "update_W!",
        (W, X, C, G, epsW) -> update_W!(W, X, C, G, epsW),
        D,
        T,
        W,
        X,
        C,
        G,
        model.epsW;
        N_runs=N_runs,
    )
end

function benchmark_update_C!(D::Int, T::Int, N_runs::Int=10)
    rng = MersenneTwister(42)
    X, P, C, W, L, G, model = create_test(D, T; rng=rng)

    return benchmark_function_with_memory(
        "update_C!", (C, X, G) -> update_C!(C, X, G), D, T, C, X, G; N_runs=N_runs
    )
end

function benchmark_update_L!(D::Int, T::Int, N_runs::Int=10)
    rng = MersenneTwister(42)
    X, P, C, W, L, G, model = create_test(D, T; rng=rng)

    return benchmark_function_with_memory(
        "update_L!", (L, P, G) -> update_L!(L, P, G), D, T, L, P, G; N_runs=N_runs
    )
end

function benchmark_calc_loss(D::Int, T::Int, N_runs::Int=10)
    rng = MersenneTwister(42)
    X, P, C, W, L, G, model = create_test(D, T; rng=rng)

    return benchmark_function_with_memory(
        "calc_loss",
        (X, P, C, W, L, G, epsC, epsW) -> calc_loss(X, P, C, W, L, G, epsC, epsW),
        D,
        T,
        X,
        P,
        C,
        W,
        L,
        G,
        model.epsC,
        model.epsW;
        N_runs=N_runs,
    )
end

# Benchmark individual function across parameter range
function benchmark_function_scaling(
    benchmark_func, params, func_name::String; N_runs::Int=10
)
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
function print_scaling_results(
    func_name::String,
    t_results::Vector{BenchmarkResult},
    d_results::Vector{BenchmarkResult},
)
    println("\n$func_name Scaling Analysis:")

    # T-scaling analysis
    t_analysis = analyze_scaling(t_results, :T)
    println(
        "  T-scaling (Fixed D=10): slope=$(Printf.@sprintf("%.2e", t_analysis.slope)) s/T, R²=$(round(t_analysis.r_squared, digits=3))",
    )

    # D-scaling analysis
    d_analysis = analyze_scaling(d_results, :D)
    println(
        "  D-scaling (Fixed T=100): slope=$(Printf.@sprintf("%.2e", d_analysis.slope)) s/D, R²=$(round(d_analysis.r_squared, digits=3))",
    )

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
        ("calc_loss", benchmark_calc_loss),
    ]

    all_results = Dict{String,Tuple{Vector{BenchmarkResult},Vector{BenchmarkResult}}}()
    all_analyses = Dict{String,Tuple{NamedTuple,NamedTuple}}()

    for (func_name, benchmark_func) in functions_to_test
        println("\n" * "="^60)

        # T-scaling benchmarks (fixed D=10)
        println("\nT-Scaling Tests (Fixed D=10):")
        t_results = benchmark_function_scaling(
            benchmark_func, T_SCALING_PARAMS, func_name; N_runs=N_runs
        )

        # D-scaling benchmarks (fixed T=100)
        println("\nD-Scaling Tests (Fixed T=100):")
        d_results = benchmark_function_scaling(
            benchmark_func, D_SCALING_PARAMS, func_name; N_runs=N_runs
        )

        # Store results
        all_results[func_name] = (t_results, d_results)

        # Analyze and print scaling
        t_analysis, d_analysis = print_scaling_results(func_name, t_results, d_results)
        all_analyses[func_name] = (t_analysis, d_analysis)
    end

    # Simplified summary table
    println("\n" * "="^70)
    println("SCALING SUMMARY")
    println("="^70)
    println(
        Printf.@sprintf(
            "%-12s | %-12s | %-5s | %-12s | %-5s",
            "Function",
            "T-Slope (s/T)",
            "T-R²",
            "D-Slope (s/D)",
            "D-R²"
        )
    )
    println("-"^70)

    for func_name in ["update_G!", "update_W!", "update_C!", "update_L!", "calc_loss"]
        t_analysis, d_analysis = all_analyses[func_name]

        println(
            Printf.@sprintf(
                "%-12s | %-12s | %-5.3f | %-12s | %-5.3f",
                func_name,
                Printf.@sprintf("%.2e", t_analysis.slope),
                t_analysis.r_squared,
                Printf.@sprintf("%.2e", d_analysis.slope),
                d_analysis.r_squared
            )
        )
    end

    # Timing summary for largest test cases
    println("\n" * "="^70)
    println("TIMING SUMMARY FOR LARGEST CASES")
    println("="^70)
    println(
        Printf.@sprintf(
            "%-12s | %-24s | %-24s", "Function", "(D=10, T=819K)", "(D=82K, T=100)"
        )
    )
    println(
        Printf.@sprintf(
            "%-12s | %-24s | %-24s", "", "Time (ms) | Mem (MB)", "Time (ms) | Mem (MB)"
        )
    )
    println("-"^70)

    for func_name in ["update_G!", "update_W!", "update_C!", "update_L!", "calc_loss"]
        t_results, d_results = all_results[func_name]

        # Find largest T case (D=10, T=1M) - should be last in t_results
        large_t_result = t_results[end]
        large_t_time = large_t_result.time_median * 1000  # Convert to ms
        large_t_mem = large_t_result.memory_allocated / 1024^2  # Convert to MB

        # Find largest D case (D=100K, T=100) - should be last in d_results
        large_d_result = d_results[end]
        large_d_time = large_d_result.time_median * 1000  # Convert to ms
        large_d_mem = large_d_result.memory_allocated / 1024^2  # Convert to MB

        println(
            Printf.@sprintf(
                "%-12s | %10.6f | %10.6f | %10.6f | %10.6f",
                func_name,
                large_t_time,
                large_t_mem,
                large_d_time,
                large_d_mem
            )
        )
    end

    # Add successive difference analysis
    println("\n" * "="^70)
    println("SUCCESSIVE DIFFERENCES")
    println("="^70)
    println(
        "Checking proportional increases in timing and memory between successive parameter values...",
    )
    println()

    for func_name in ["update_G!", "update_W!", "update_C!", "update_L!", "calc_loss"]
        t_results, d_results = all_results[func_name]

        println("$func_name:")

        # T-scaling successive analysis
        println("  T-scaling (Fixed D=10):")
        for i in 2:length(t_results)
            prev = t_results[i - 1]
            curr = t_results[i]

            param_ratio = curr.T / prev.T
            time_ratio = curr.time_median / prev.time_median
            mem_ratio = curr.memory_allocated / prev.memory_allocated

            println(
                Printf.@sprintf(
                    "    T: %d→%d (×%.1f) | Time: ×%.2f | Mem: ×%.2f",
                    prev.T,
                    curr.T,
                    param_ratio,
                    time_ratio,
                    mem_ratio
                )
            )
        end

        # D-scaling successive analysis
        println("  D-scaling (Fixed T=100):")
        for i in 2:length(d_results)
            prev = d_results[i - 1]
            curr = d_results[i]

            param_ratio = curr.D / prev.D
            time_ratio = curr.time_median / prev.time_median
            mem_ratio = curr.memory_allocated / prev.memory_allocated

            println(
                Printf.@sprintf(
                    "    D: %d→%d (×%.1f) | Time: ×%.2f | Mem: ×%.2f",
                    prev.D,
                    curr.D,
                    param_ratio,
                    time_ratio,
                    mem_ratio
                )
            )
        end
        println()
    end

    # Save results to JSON file
    save_results_to_json(all_results, all_analyses)

    return all_results, all_analyses
end

# Function to save benchmark results to JSON
function save_results_to_json(all_results, all_analyses)
    println("\nSaving benchmark data to JSON...")

    # Create output directory if it doesn't exist
    output_dir = "benchmark_results"
    if !isdir(output_dir)
        mkdir(output_dir)
        println("Created directory: $output_dir/")
    end

    # Prepare data structure for JSON export
    export_data = Dict{String,Any}()

    # Add metadata
    export_data["metadata"] = Dict(
        "timestamp" => string(now()),
        "julia_version" => string(VERSION),
        "description" => "eSPA core functions scaling benchmark results",
        "T_range" => "100 to 819,200 (fixed D=10)",
        "D_range" => "10 to 81,920 (fixed T=100)",
    )

    # Add raw benchmark results
    export_data["raw_results"] = Dict{String,Any}()
    for (func_name, (t_results, d_results)) in all_results
        export_data["raw_results"][func_name] = Dict(
            "T_scaling" => [
                Dict(
                    "D" => r.D,
                    "T" => r.T,
                    "time_median_seconds" => r.time_median,
                    "time_std_seconds" => r.time_std,
                    "memory_allocated_bytes" => r.memory_allocated,
                    "all_times_seconds" => r.times,
                ) for r in t_results
            ],
            "D_scaling" => [
                Dict(
                    "D" => r.D,
                    "T" => r.T,
                    "time_median_seconds" => r.time_median,
                    "time_std_seconds" => r.time_std,
                    "memory_allocated_bytes" => r.memory_allocated,
                    "all_times_seconds" => r.times,
                ) for r in d_results
            ],
        )
    end

    # Add scaling analysis results
    export_data["scaling_analysis"] = Dict{String,Any}()
    for (func_name, (t_analysis, d_analysis)) in all_analyses
        export_data["scaling_analysis"][func_name] = Dict(
            "T_scaling" => Dict(
                "slope_seconds_per_T" => t_analysis.slope,
                "intercept_seconds" => t_analysis.intercept,
                "r_squared" => t_analysis.r_squared,
            ),
            "D_scaling" => Dict(
                "slope_seconds_per_D" => d_analysis.slope,
                "intercept_seconds" => d_analysis.intercept,
                "r_squared" => d_analysis.r_squared,
            ),
        )
    end

    # Add successive ratios analysis
    export_data["successive_ratios"] = Dict{String,Any}()
    for (func_name, (t_results, d_results)) in all_results
        t_ratios = []
        for i in 2:length(t_results)
            prev, curr = t_results[i - 1], t_results[i]
            push!(
                t_ratios,
                Dict(
                    "from_T" => prev.T,
                    "to_T" => curr.T,
                    "parameter_ratio" => curr.T / prev.T,
                    "time_ratio" => curr.time_median / prev.time_median,
                    "memory_ratio" => curr.memory_allocated / prev.memory_allocated,
                ),
            )
        end

        d_ratios = []
        for i in 2:length(d_results)
            prev, curr = d_results[i - 1], d_results[i]
            push!(
                d_ratios,
                Dict(
                    "from_D" => prev.D,
                    "to_D" => curr.D,
                    "parameter_ratio" => curr.D / prev.D,
                    "time_ratio" => curr.time_median / prev.time_median,
                    "memory_ratio" => curr.memory_allocated / prev.memory_allocated,
                ),
            )
        end

        export_data["successive_ratios"][func_name] = Dict(
            "T_scaling" => t_ratios, "D_scaling" => d_ratios
        )
    end

    # Generate filename
    filename = joinpath(output_dir, "espa_benchmark_results.json")

    # Write to JSON file
    open(filename, "w") do io
        JSON3.pretty(io, export_data)
    end

    return println("Benchmark data saved to: $filename")
end

# Run the benchmark if called directly
if abspath(PROGRAM_FILE) == @__FILE__
    results, analyses = run_scaling_benchmarks(; N_runs=10)

    println("\nScaling benchmark complete!")
end
