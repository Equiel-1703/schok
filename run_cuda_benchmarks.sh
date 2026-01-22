#!/bin/bash

RUNS_PER_BENCHMARK=30 # CHANGE TO 30 LATER ----------------------------------------------------<<<

# Define directories for benchmarks
BENCHMARKS_DIR="benchmarks"
CUDA_BENCHMARKS_DIR="$BENCHMARKS_DIR/cuda"

# --- Functions for running benchmarks ---
run_CUDA_benchmark() {
    local benchmark_name="$1"
    local benchmark_input="$2"
    
    local output_name="${benchmark_name%.cu}.out"
    
    # Check if the compiled file does not exist
    if [ ! -f "$CUDA_BENCHMARKS_DIR/$output_name" ]; then
        # Compile the CUDA benchmark if it does not exist
        nvcc -o "$CUDA_BENCHMARKS_DIR/$output_name" "$CUDA_BENCHMARKS_DIR/$benchmark_name"
    fi
    
    local i # Loop variable
    
    # Run the compiled CUDA benchmark
    for ((i=1; i<=RUNS_PER_BENCHMARK; i++)); do
        "./$CUDA_BENCHMARKS_DIR/$output_name" $benchmark_input 2>&1
    done
}


# ------------------ Script Start ------------------
echo -e "- CUDA Benchmarks Results -"
date
echo -e "Tests conducted by: Andre R. Du Bois & Henrique G. Rodrigues\n"
echo -e "Runs per benchmark: $RUNS_PER_BENCHMARK\n"
echo ""  # Add a blank line for readability

# ------------------ Dot Product Benchmark ------------------
# INPUTS="1024 2048 4096" # Test values
INPUTS="400000000 500000000 600000000" # CHANGE TO THIS LATER ----------------------------------------------------<<<

echo -e "Dot Product (DP) benchmark\n"

BENCH="dot_product.cu"
for INPUT in $INPUTS; do
    run_CUDA_benchmark "$BENCH" "$INPUT"
done
echo ""

# ------------------ Julia Benchmark ------------------
# INPUTS="512 1024 2048" # Test values
INPUTS="7168 9216 11264" # CHANGE TO THIS LATER ----------------------------------------------------<<<

echo -e "Julia (JL) benchmark\n"

BENCH="julia.cu"
for INPUT in $INPUTS; do
    run_CUDA_benchmark "$BENCH" "$INPUT"
done
echo ""

# ------------------ MM Benchmarks ------------------
# INPUTS="128 256 512" # Test values
INPUTS="5000 7000 9000" # CHANGE TO THIS LATER ----------------------------------------------------<<<

echo -e "Matrix Multiplication (MM) benchmark\n"

BENCH="mm.cu"
for INPUT in $INPUTS; do
    run_CUDA_benchmark "$BENCH" "$INPUT"
done
echo ""

# ------------------ NBody Benchmarks ------------------
# INPUTS="128 256 512" # Test values
INPUTS="100000 200000 400000" # CHANGE TO THIS LATER ----------------------------------------------------<<<

echo -e "nBodies (NB) benchmark\n"

BENCH="nbodies.cu"
for INPUT in $INPUTS; do
    run_CUDA_benchmark "$BENCH" "$INPUT"
done
echo ""

# ------------------ Nearest Neighbor Benchmarks ------------------
# INPUTS="1024 2048 4096" # Test values
INPUTS="100000000 200000000 300000000" # CHANGE TO THIS LATER ----------------------------------------------------<<<

echo -e "Nearest Neighbor (NN) benchmark\n"

BENCH="nearest_neighbor_double.cu"
for INPUT in $INPUTS; do
    run_CUDA_benchmark "$BENCH" "$INPUT"
done
echo ""

# ------------------ Raytracer Benchmarks ------------------
# INPUTS="512 1024 2048" # Test values
INPUTS="7168 9216 11264" # CHANGE TO THIS LATER ----------------------------------------------------<<<

echo -e "Raytracer (RT) benchmark\n"

BENCH="raytracer.cu"
for INPUT in $INPUTS; do
    run_CUDA_benchmark "$BENCH" "$INPUT"
done
echo ""

# ------------------ Script End ------------------