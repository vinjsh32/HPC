# High-Performance OBDD Library - Developer Documentation

/**
 * @file CLAUDE.md
 * @brief Comprehensive Developer Guide for High-Performance OBDD Library
 * 
 * This documentation provides complete guidance for development, testing,
 * and performance analysis of the multi-backend OBDD library. The guide
 * covers build procedures, backend selection, performance benchmarking,
 * and empirical analysis methodologies.
 * 
 * @author @vijsh32
 * @date August 17, 2024
 * @version 2.1
 * @copyright 2024 High Performance Computing Laboratory
 */

This file provides comprehensive guidance for developers working with the
high-performance OBDD library codebase.

## Project Overview

This is an Ordered Binary Decision Diagram (OBDD) library with support for sequential CPU, OpenMP parallel, and CUDA GPU backends. The library provides a C/C++ interface for building and manipulating Boolean functions represented as OBDDs.

## Build Commands

The build system uses a Makefile with configurable backends:

```bash
# Sequential CPU only
make CUDA=0

# OpenMP parallel backend  
make OMP=1 CUDA=0

# CUDA GPU backend (default)
make CUDA=1

# Debug builds
make DEBUG=1

# Clean build artifacts
make clean
```

## Test Commands

Run tests using make targets:

```bash
# Sequential CPU test
make CUDA=0 run-seq

# OpenMP parallel test  
make OMP=1 CUDA=0 run-omp

# CUDA GPU test
make CUDA=1 run-cuda

# Advanced variable reordering algorithms test
make run-reordering

# Advanced mathematical problems test
make run-advmath

# Comprehensive performance benchmarks (NEW)
make run-benchmark
```

## Performance Benchmarking

The library includes a comprehensive performance benchmark suite for comparing Sequential CPU, OpenMP Parallel, and CUDA GPU backends:

### Benchmark Features
- **Multi-Backend Comparison**: Automatic testing across all available backends
- **Scalability Analysis**: Performance vs problem size analysis
- **Memory Usage Profiling**: Detailed memory consumption metrics
- **Throughput Measurement**: Operations per second and node processing rates
- **Report Generation**: CSV, JSON, and formatted text reports

### Running Benchmarks
```bash
# Run comprehensive performance comparison
make run-benchmark

# Results are saved to benchmark_results.csv for analysis
```

### Benchmark Metrics
- **Execution Time**: Wall-clock time for operations
- **Memory Usage**: Peak memory consumption
- **Throughput**: Operations and nodes processed per second
- **Scalability**: Performance vs problem size
- **Parallel Efficiency**: Speedup relative to number of cores
- **Resource Utilization**: CPU and GPU utilization percentages

Tests are built on GoogleTest framework and located in `tests/` directory.

## Advanced Variable Reordering

The library includes state-of-the-art algorithms for optimal variable ordering:

### Algorithms Available
- **Sifting Algorithm**: Moves each variable to its locally optimal position
- **Window Permutation with DP**: Exhaustive search within sliding windows using dynamic programming
- **Simulated Annealing**: Global optimization with probabilistic acceptance of worse solutions  
- **Genetic Algorithm**: Evolutionary approach with tournament selection and order-preserving crossover
- **Hybrid Strategy**: Combines multiple algorithms for best results

### Usage Example
```cpp
#include "obdd_reordering.hpp"

// Create BDD
OBDD* bdd = obdd_create(num_vars, initial_order);
bdd->root = /* construct your BDD */;

// Configure reordering algorithm
ReorderConfig config = obdd_reorder_get_default_config(REORDER_GENETIC);
config.population_size = 50;
config.max_iterations = 20;

// Apply reordering
ReorderResult result = {};
int* optimized_order = obdd_reorder_advanced(bdd, &config, &result);

// Print results
obdd_print_reorder_result(&result);

// Cleanup
free(optimized_order);
```

### Performance Features
- **OpenMP parallelization** for Sifting and Genetic algorithms
- **Memoization** in Window Permutation to avoid redundant computations
- **Configurable parameters** for all algorithms
- **Performance metrics** tracking (time, iterations, swaps, reduction ratio)

## Advanced Mathematical Applications

The library includes complex mathematical problem encodings as OBDD constraints:

### Problem Categories
- **Modular Arithmetic**: x² + y² ≡ z² (mod p), discrete logarithms, modular multiplication
- **Cryptographic Functions**: AES S-box, SHA-1 choice/majority functions, DES S-boxes
- **Diophantine Equations**: Linear equations (ax + by = c), Pell equations (x² - Dy² = 1), Pythagorean triples
- **Combinatorial Problems**: N-Queens, graph 3-coloring, Hamiltonian paths, knapsack problems
- **Boolean Satisfiability**: CNF formulas, random 3-SAT instances, Sudoku puzzles

### Usage Example
```cpp
#include "obdd_advanced_math.hpp"

// Solve modular arithmetic constraint
OBDD* pythagorean_mod = obdd_modular_pythagorean(4, 7); // 4 bits, mod 7

// Encode AES S-box as BDD
OBDD* aes_sbox = obdd_aes_sbox();

// Solve N-Queens problem
OBDD* queens = obdd_n_queens(8); // 8x8 chessboard

// Run comprehensive benchmarks
AdvancedBenchmark results[10];
int num_benchmarks = obdd_run_advanced_benchmarks(results, 10);
obdd_print_benchmark_results(results, num_benchmarks);
```

### Research Applications
- **Formal verification** of cryptographic protocols
- **Constraint solving** for combinatorial optimization
- **Model checking** of complex systems
- **Automated theorem proving** for mathematical conjectures

## Architecture

### Core Components

- **obdd.hpp/obdd.h**: Public C/C++ API with `extern "C"` linkage
- **obdd_core.cpp**: Sequential implementation of core OBDD operations
- **obdd_openmp.cpp/obdd_openmp_optim.cpp**: OpenMP parallel implementations
- **obdd_cuda.cu**: CUDA GPU implementations using Thrust

### Key Data Structures

- `OBDDNode`: Basic BDD node with variable index, high/low children, and reference count
- `OBDD`: Handle containing root node, variable count, and variable ordering
- Global unique table and apply cache for memoization (thread-safe with mutex)

### Memory Management

All dynamically created nodes are tracked in a global set for cleanup. Constant leaves (0/1) are singletons. Thread safety is ensured via global mutex for node creation/destruction.

### Backend Selection

Backends are selected via compile-time flags:
- `OBDD_ENABLE_OPENMP`: Enables OpenMP parallel operations
- `OBDD_ENABLE_CUDA`: Enables CUDA GPU operations

### GPU Architecture Detection

The build system automatically detects GPU compute capability via `scripts/detect_gpu_arch.sh` which queries `nvidia-smi` and sets appropriate NVCC flags.

### Variable Reordering

- CPU: Parallel merge sort via OpenMP
- GPU: Thrust-based sorting on device

## Important Implementation Details

- Per-thread apply caches in OpenMP backend reduce lock contention
- CUDA operations copy data to/from device and return reduced OBDDs
- All public APIs maintain C linkage for compatibility
- Reference counting prevents premature node deallocation