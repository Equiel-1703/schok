#!/bin/bash

RUNS_PER_BENCHMARK=30 # CHANGE TO 30 LATER ----------------------------------------------------<<<

# Define directories for benchmarks
BENCHMARKS_DIR="benchmarks/cola"

# --- Functions for running benchmarks ---
run_elixir_benchmark() {
    local benchmark_name="$1"
    local benchmark_input="$2"
    
    local i # Loop variable
    
    for ((i=1; i<=RUNS_PER_BENCHMARK; i++)); do
        mix run $BENCHMARKS_DIR/$benchmark_name $benchmark_input 2>&1
    done
}

# ------------------ Script Start ------------------
echo -e "- Schok Benchmarks Results -"
date
echo -e "Tests conducted by: Andre R. Du Bois & Henrique G. Rodrigues\n"
echo -e "Runs per benchmark: $RUNS_PER_BENCHMARK\n"
echo ""  # Add a blank line for readability

# ------------------ Dot Product Benchmark ------------------
INPUTS="1024 2048 4096" # Test values
# INPUTS="400000000 500000000 600000000" # CHANGE TO THIS LATER ----------------------------------------------------<<<

echo -e "Dot Product (DP) benchmark\n"

BENCH="dot_product.ex"
for INPUT in $INPUTS; do
    run_elixir_benchmark "$BENCH" "$INPUT"
done
echo ""

# ------------------ Julia Benchmark ------------------
INPUTS="512 1024 2048" # Test values
# INPUTS="7168 9216 11264" # CHANGE TO THIS LATER ----------------------------------------------------<<<

echo -e "Julia (JL) benchmark\n"

BENCH="julia.ex"
for INPUT in $INPUTS; do
    run_elixir_benchmark "$BENCH" "$INPUT"
done
echo ""

# ------------------ MM Benchmarks ------------------
INPUTS="128 256 512" # Test values
# INPUTS="5000 7000 9000" # CHANGE TO THIS LATER ----------------------------------------------------<<<

echo -e "Matrix Multiplication (MM) benchmark\n"

BENCH="mm.ex"
for INPUT in $INPUTS; do
    run_elixir_benchmark "$BENCH" "$INPUT"
done
echo ""

# ------------------ NBody Benchmarks ------------------
INPUTS="128 256 512" # Test values
# INPUTS="100000 200000 400000" # CHANGE TO THIS LATER ----------------------------------------------------<<<

echo -e "nBodies (NB) benchmark\n"

BENCH="nbodies.ex"
for INPUT in $INPUTS; do
    run_elixir_benchmark "$BENCH" "$INPUT"
done
echo ""

# ------------------ Nearest Neighbor Benchmarks ------------------
INPUTS="1024 2048 4096" # Test values
# INPUTS="100000000 200000000 300000000" # CHANGE TO THIS LATER ----------------------------------------------------<<<

echo -e "Nearest Neighbor (NN) benchmark\n"

BENCH="nearest_neighbor.ex"
for INPUT in $INPUTS; do
    run_elixir_benchmark "$BENCH" "$INPUT"
done
echo ""

# ------------------ Raytracer Benchmarks ------------------
INPUTS="512 1024 2048" # Test values
# INPUTS="7168 9216 11264" # CHANGE TO THIS LATER ----------------------------------------------------<<<

echo -e "Raytracer (RT) benchmark\n"

BENCH="raytracer.ex"
for INPUT in $INPUTS; do
    run_elixir_benchmark "$BENCH" "$INPUT"
done
echo ""

# ------------------ Saxpy Benchmarks ------------------
INPUTS="512 1024 2048" # Test values
# INPUTS="300000000 400000000 500000000" # CHANGE TO THIS LATER ----------------------------------------------------<<<

echo -e "Saxpy (SP) benchmark\n"

BENCH="saxpy.ex"
for INPUT in $INPUTS; do
    run_elixir_benchmark "$BENCH" "$INPUT"
done
echo ""

# ------------------ Script End ------------------
