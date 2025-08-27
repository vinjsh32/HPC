/**
 * @file performance_benchmark.hpp
 * @brief Comprehensive performance benchmark suite for OBDD backend comparison
 * @version 1.0
 * @date 2024
 * 
 * This module provides detailed performance benchmarking capabilities to compare
 * Sequential CPU, OpenMP Parallel, and CUDA GPU backends across multiple metrics:
 * - Execution time analysis
 * - Memory usage profiling  
 * - Scalability testing
 * - Throughput measurements
 * - Energy efficiency analysis
 * 
 * @author HPC Team
 * @copyright 2024 High Performance Computing Laboratory
 */

#pragma once

#include "core/obdd.hpp"
#include <chrono>
#include <vector>
#include <string>

#ifdef OBDD_ENABLE_CUDA
#include "cuda/obdd_cuda.hpp"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Backend types for performance comparison
 */
typedef enum {
    BACKEND_SEQUENTIAL = 0,
    BACKEND_OPENMP = 1,
    BACKEND_CUDA = 2
} BackendType;

/**
 * @brief Performance metrics for a single benchmark test
 */
typedef struct {
    BackendType backend;
    char test_name[64];
    
    // Timing metrics
    double execution_time_ms;
    double setup_time_ms;
    double cleanup_time_ms;
    
    // Memory metrics
    size_t peak_memory_usage_bytes;
    size_t final_bdd_nodes;
    size_t intermediate_nodes_created;
    
    // Throughput metrics
    double operations_per_second;
    double nodes_processed_per_second;
    
    // Scalability metrics
    int num_variables;
    int problem_complexity;
    double parallel_efficiency;  // Only for OpenMP/CUDA
    
    // Quality metrics
    int result_correctness;      // 1 if correct, 0 if incorrect
    double numerical_accuracy;   // For mathematical problems
    
    // Resource utilization
    double cpu_utilization_percent;
    double memory_bandwidth_gbps;
    int gpu_sm_utilization_percent; // CUDA only
    
} BenchmarkResult;

/**
 * @brief Comprehensive benchmark suite configuration
 */
typedef struct {
    int min_variables;
    int max_variables;
    int variable_step;
    
    int num_repetitions;
    int warmup_iterations;
    
    int enable_memory_profiling;
    int enable_energy_measurement;
    int enable_detailed_timing;
    
    double timeout_seconds;
    
} BenchmarkConfig;

/**
 * @brief Benchmark test categories
 */
typedef enum {
    TEST_BASIC_OPERATIONS = 0,  // AND, OR, NOT, XOR
    TEST_COMPLEX_FUNCTIONS,     // Multi-level Boolean functions
    TEST_VARIABLE_REORDERING,   // Reordering algorithm performance
    TEST_MATHEMATICAL_ENCODING, // Cryptographic and combinatorial problems
    TEST_SCALABILITY,           // Large problem instances
    TEST_MEMORY_INTENSIVE,      // Memory-bound operations
    TEST_COMPUTE_INTENSIVE,     // Computation-bound operations
    TEST_CACHE_BEHAVIOR         // Cache efficiency testing
} BenchmarkTestType;

/**
 * @brief Individual test case definition
 */
typedef struct {
    BenchmarkTestType type;
    char name[64];
    char description[256];
    
    int num_variables;
    int complexity_level;
    
    // Function pointer for test execution
    OBDD* (*setup_function)(int variables, int complexity);
    int (*execute_function)(OBDD* bdd, BackendType backend);
    void (*cleanup_function)(OBDD* bdd);
    
} BenchmarkTestCase;

// =====================================================
// Benchmark Suite Functions
// =====================================================

/**
 * @brief Initialize benchmark suite with default configuration
 * @return Default benchmark configuration
 */
BenchmarkConfig benchmark_get_default_config(void);

/**
 * @brief Run comprehensive performance comparison across all backends
 * @param config Benchmark configuration
 * @param results Array to store benchmark results
 * @param max_results Maximum number of results to store
 * @return Number of benchmark results generated
 */
int benchmark_run_comprehensive(const BenchmarkConfig* config, 
                               BenchmarkResult* results, 
                               int max_results);

/**
 * @brief Run specific test case across all available backends
 * @param test_case Test case to execute
 * @param results Array to store results (one per backend)
 * @return Number of results generated
 */
int benchmark_run_test_case(const BenchmarkTestCase* test_case, 
                           BenchmarkResult* results);

/**
 * @brief Compare two backends on specific problem size
 * @param backend1 First backend to compare
 * @param backend2 Second backend to compare  
 * @param num_variables Problem size
 * @param test_type Type of test to perform
 * @param result1 Result for first backend
 * @param result2 Result for second backend
 * @return 0 on success, -1 on failure
 */
int benchmark_compare_backends(BackendType backend1, BackendType backend2,
                              int num_variables, BenchmarkTestType test_type,
                              BenchmarkResult* result1, BenchmarkResult* result2);

// =====================================================
// Test Case Generators
// =====================================================

/**
 * @brief Generate basic operation test cases
 * @param num_variables Number of variables
 * @param complexity Complexity level (1-5)
 * @return Generated BDD for testing
 */
OBDD* benchmark_generate_basic_operations(int num_variables, int complexity);

/**
 * @brief Generate complex Boolean function test cases
 * @param num_variables Number of variables
 * @param complexity Complexity level (1-5)
 * @return Generated BDD for testing
 */
OBDD* benchmark_generate_complex_function(int num_variables, int complexity);

/**
 * @brief Generate scalability test cases
 * @param num_variables Number of variables
 * @param complexity Complexity level (1-5)
 * @return Generated BDD for testing
 */
OBDD* benchmark_generate_scalability_test(int num_variables, int complexity);

/**
 * @brief Generate memory-intensive test cases
 * @param num_variables Number of variables
 * @param complexity Complexity level (1-5)
 * @return Generated BDD for testing
 */
OBDD* benchmark_generate_memory_intensive(int num_variables, int complexity);

// =====================================================
// Performance Analysis Functions
// =====================================================

/**
 * @brief Analyze performance results and generate statistics
 * @param results Array of benchmark results
 * @param num_results Number of results
 * @param analysis_output Buffer for analysis text
 * @param buffer_size Size of analysis buffer
 */
void benchmark_analyze_results(const BenchmarkResult* results, int num_results,
                              char* analysis_output, size_t buffer_size);

/**
 * @brief Calculate speedup between backends
 * @param baseline_result Baseline backend result
 * @param comparison_result Comparison backend result
 * @return Speedup factor (>1 means faster, <1 means slower)
 */
double benchmark_calculate_speedup(const BenchmarkResult* baseline_result,
                                  const BenchmarkResult* comparison_result);

/**
 * @brief Calculate parallel efficiency for multi-core backends
 * @param sequential_result Sequential execution result
 * @param parallel_result Parallel execution result
 * @param num_cores Number of cores used
 * @return Parallel efficiency (0.0 to 1.0)
 */
double benchmark_calculate_efficiency(const BenchmarkResult* sequential_result,
                                     const BenchmarkResult* parallel_result,
                                     int num_cores);

// =====================================================
// Report Generation Functions
// =====================================================

/**
 * @brief Generate detailed performance report in text format
 * @param results Array of benchmark results
 * @param num_results Number of results
 * @param output_file Path to output file
 * @return 0 on success, -1 on failure
 */
int benchmark_generate_text_report(const BenchmarkResult* results, int num_results,
                                  const char* output_file);

/**
 * @brief Generate performance report in CSV format for analysis
 * @param results Array of benchmark results
 * @param num_results Number of results
 * @param output_file Path to output CSV file
 * @return 0 on success, -1 on failure
 */
int benchmark_generate_csv_report(const BenchmarkResult* results, int num_results,
                                 const char* output_file);

/**
 * @brief Generate performance report in JSON format
 * @param results Array of benchmark results
 * @param num_results Number of results
 * @param output_file Path to output JSON file
 * @return 0 on success, -1 on failure
 */
int benchmark_generate_json_report(const BenchmarkResult* results, int num_results,
                                  const char* output_file);

/**
 * @brief Print benchmark results to console in formatted table
 * @param results Array of benchmark results
 * @param num_results Number of results
 */
void benchmark_print_results(const BenchmarkResult* results, int num_results);

/**
 * @brief Print performance comparison summary
 * @param results Array of benchmark results
 * @param num_results Number of results
 */
void benchmark_print_comparison_summary(const BenchmarkResult* results, int num_results);

// =====================================================
// Utility Functions
// =====================================================

/**
 * @brief Get backend name as string
 * @param backend Backend type
 * @return Backend name string
 */
const char* benchmark_get_backend_name(BackendType backend);

/**
 * @brief Check if backend is available on current system
 * @param backend Backend type to check
 * @return 1 if available, 0 if not available
 */
int benchmark_is_backend_available(BackendType backend);

/**
 * @brief Get system information for benchmark context
 * @param info_buffer Buffer for system information
 * @param buffer_size Size of info buffer
 */
void benchmark_get_system_info(char* info_buffer, size_t buffer_size);

/**
 * @brief Validate benchmark result for correctness
 * @param result Benchmark result to validate
 * @return 1 if valid, 0 if invalid
 */
int benchmark_validate_result(const BenchmarkResult* result);

#ifdef __cplusplus
}
#endif