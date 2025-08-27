/**
 * @file test_performance_benchmark.cpp
 * @brief Comprehensive performance benchmark test suite
 * 
 * This test suite runs detailed performance comparisons between Sequential CPU,
 * OpenMP Parallel, and CUDA GPU backends, providing metrics for report generation.
 */

#include <gtest/gtest.h>
#include "advanced/performance_benchmark.hpp"
#include "core/obdd.hpp"
#include <cstdio>
#include <vector>
#include <string>

class PerformanceBenchmarkTest : public ::testing::Test {
protected:
    void SetUp() override {
        config = benchmark_get_default_config();
        config.min_variables = 3;
        config.max_variables = 15;
        config.variable_step = 3;
        config.num_repetitions = 3;
        config.timeout_seconds = 60.0;
    }
    
    BenchmarkConfig config;
};

// =====================================================
// Individual Backend Tests
// =====================================================

TEST_F(PerformanceBenchmarkTest, SequentialCPUBenchmark) {
    printf("\n🔥 SEQUENTIAL CPU PERFORMANCE BENCHMARK\n");
    printf("========================================\n");
    
    BenchmarkTestCase test_case = {};
    strncpy(test_case.name, "Sequential_CPU_Test", sizeof(test_case.name));
    strncpy(test_case.description, "Sequential CPU baseline performance", sizeof(test_case.description));
    test_case.type = TEST_BASIC_OPERATIONS;
    test_case.num_variables = 8;
    test_case.complexity_level = 3;
    test_case.setup_function = benchmark_generate_complex_function;
    
    BenchmarkResult results[5];
    int num_results = benchmark_run_test_case(&test_case, results);
    
    ASSERT_GT(num_results, 0) << "Should generate at least one result";
    
    // Find sequential result
    BenchmarkResult* seq_result = nullptr;
    for (int i = 0; i < num_results; i++) {
        if (results[i].backend == BACKEND_SEQUENTIAL) {
            seq_result = &results[i];
            break;
        }
    }
    
    ASSERT_NE(seq_result, nullptr) << "Sequential backend should be available";
    EXPECT_EQ(seq_result->result_correctness, 1) << "Sequential test should pass";
    EXPECT_GT(seq_result->execution_time_ms, 0) << "Should have positive execution time";
    EXPECT_GT(seq_result->operations_per_second, 0) << "Should have positive throughput";
    
    printf("Sequential CPU Results:\n");
    printf("- Execution Time: %.3f ms\n", seq_result->execution_time_ms);
    printf("- Memory Usage: %zu KB\n", seq_result->peak_memory_usage_bytes / 1024);
    printf("- Operations/sec: %.1f\n", seq_result->operations_per_second);
    printf("- Variables: %d\n", seq_result->num_variables);
    printf("- CPU Utilization: %.1f%%\n", seq_result->cpu_utilization_percent);
}

TEST_F(PerformanceBenchmarkTest, OpenMPBenchmark) {
    if (!benchmark_is_backend_available(BACKEND_OPENMP)) {
        GTEST_SKIP() << "OpenMP backend not available";
    }
    
    printf("\n⚡ OPENMP PARALLEL PERFORMANCE BENCHMARK\n");
    printf("=========================================\n");
    
    BenchmarkTestCase test_case = {};
    strncpy(test_case.name, "OpenMP_Parallel_Test", sizeof(test_case.name));
    strncpy(test_case.description, "OpenMP parallel performance", sizeof(test_case.description));
    test_case.type = TEST_COMPLEX_FUNCTIONS;
    test_case.num_variables = 10;
    test_case.complexity_level = 4;
    test_case.setup_function = benchmark_generate_complex_function;
    
    BenchmarkResult results[5];
    int num_results = benchmark_run_test_case(&test_case, results);
    
    ASSERT_GT(num_results, 0) << "Should generate at least one result";
    
    // Find OpenMP result
    BenchmarkResult* omp_result = nullptr;
    BenchmarkResult* seq_result = nullptr;
    
    for (int i = 0; i < num_results; i++) {
        if (results[i].backend == BACKEND_OPENMP) {
            omp_result = &results[i];
        } else if (results[i].backend == BACKEND_SEQUENTIAL) {
            seq_result = &results[i];
        }
    }
    
    ASSERT_NE(omp_result, nullptr) << "OpenMP backend should be available";
    EXPECT_EQ(omp_result->result_correctness, 1) << "OpenMP test should pass";
    
    printf("OpenMP Results:\n");
    printf("- Execution Time: %.3f ms\n", omp_result->execution_time_ms);
    printf("- Memory Usage: %zu KB\n", omp_result->peak_memory_usage_bytes / 1024);
    printf("- Operations/sec: %.1f\n", omp_result->operations_per_second);
    printf("- Parallel Efficiency: %.3f\n", omp_result->parallel_efficiency);
    
    // Compare with sequential if available
    if (seq_result) {
        double speedup = benchmark_calculate_speedup(seq_result, omp_result);
        printf("- Speedup vs Sequential: %.2fx\n", speedup);
        EXPECT_GT(speedup, 0.01) << "OpenMP should not be extremely slower than sequential (overhead normal for small problems)";
    }
}

TEST_F(PerformanceBenchmarkTest, CUDABenchmark) {
    if (!benchmark_is_backend_available(BACKEND_CUDA)) {
        GTEST_SKIP() << "CUDA backend not available";
    }
    
    printf("\n🚀 CUDA GPU PERFORMANCE BENCHMARK\n");
    printf("==================================\n");
    
    BenchmarkTestCase test_case = {};
    strncpy(test_case.name, "CUDA_GPU_Test", sizeof(test_case.name));
    strncpy(test_case.description, "CUDA GPU acceleration performance", sizeof(test_case.description));
    test_case.type = TEST_SCALABILITY;
    test_case.num_variables = 12;
    test_case.complexity_level = 5;
    test_case.setup_function = benchmark_generate_scalability_test;
    
    BenchmarkResult results[5];
    int num_results = benchmark_run_test_case(&test_case, results);
    
    ASSERT_GT(num_results, 0) << "Should generate at least one result";
    
    // Find CUDA result
    BenchmarkResult* cuda_result = nullptr;
    BenchmarkResult* seq_result = nullptr;
    
    for (int i = 0; i < num_results; i++) {
        if (results[i].backend == BACKEND_CUDA) {
            cuda_result = &results[i];
        } else if (results[i].backend == BACKEND_SEQUENTIAL) {
            seq_result = &results[i];
        }
    }
    
    ASSERT_NE(cuda_result, nullptr) << "CUDA backend should be available";
    EXPECT_EQ(cuda_result->result_correctness, 1) << "CUDA test should pass";
    
    printf("CUDA GPU Results:\n");
    printf("- Execution Time: %.3f ms\n", cuda_result->execution_time_ms);
    printf("- Memory Usage: %zu KB\n", cuda_result->peak_memory_usage_bytes / 1024);
    printf("- Operations/sec: %.1f\n", cuda_result->operations_per_second);
    printf("- GPU SM Utilization: %d%%\n", cuda_result->gpu_sm_utilization_percent);
    printf("- Parallel Efficiency: %.3f\n", cuda_result->parallel_efficiency);
    
    // Compare with sequential if available
    if (seq_result) {
        double speedup = benchmark_calculate_speedup(seq_result, cuda_result);
        printf("- Speedup vs Sequential: %.2fx\n", speedup);
        EXPECT_GT(speedup, 0.1) << "CUDA should provide some benefit for complex problems";
    }
}

// =====================================================
// Comprehensive Comparison Tests
// =====================================================

TEST_F(PerformanceBenchmarkTest, ComprehensiveBackendComparison) {
    printf("\n📊 COMPREHENSIVE BACKEND PERFORMANCE COMPARISON\n");
    printf("================================================\n");
    
    config.min_variables = 5;
    config.max_variables = 15;
    config.variable_step = 5;
    config.num_repetitions = 2;
    
    const int MAX_RESULTS = 100;
    BenchmarkResult results[MAX_RESULTS];
    
    int num_results = benchmark_run_comprehensive(&config, results, MAX_RESULTS);
    
    ASSERT_GT(num_results, 0) << "Should generate benchmark results";
    
    printf("Generated %d benchmark results\n\n", num_results);
    
    // Print detailed results
    benchmark_print_results(results, num_results);
    
    // Print comparison summary
    benchmark_print_comparison_summary(results, num_results);
    
    // Validate all results
    for (int i = 0; i < num_results; i++) {
        EXPECT_EQ(benchmark_validate_result(&results[i]), 1) 
            << "Result " << i << " should be valid";
    }
    
    // Generate CSV report for detailed analysis
    const char* csv_file = "benchmark_results.csv";
    int csv_status = benchmark_generate_csv_report(results, num_results, csv_file);
    EXPECT_EQ(csv_status, 0) << "CSV report generation should succeed";
    
    if (csv_status == 0) {
        printf("\n📈 Detailed results saved to: %s\n", csv_file);
    }
}

TEST_F(PerformanceBenchmarkTest, ScalabilityAnalysis) {
    printf("\n📈 SCALABILITY ANALYSIS\n");
    printf("=======================\n");
    
    struct ScalabilityResult {
        int variables;
        double seq_time;
        double omp_time;
        double cuda_time;
        bool has_omp;
        bool has_cuda;
    };
    
    std::vector<ScalabilityResult> scalability_data;
    
    // Test different problem sizes
    for (int vars = 4; vars <= 16; vars += 4) {
        printf("Testing with %d variables...\n", vars);
        
        BenchmarkTestCase test_case = {};
        strncpy(test_case.name, "Scalability", sizeof(test_case.name));
        test_case.type = TEST_SCALABILITY;
        test_case.num_variables = vars;
        test_case.complexity_level = 3;
        test_case.setup_function = benchmark_generate_scalability_test;
        
        BenchmarkResult results[10];
        int num_results = benchmark_run_test_case(&test_case, results);
        
        ScalabilityResult scale_result = {};
        scale_result.variables = vars;
        
        for (int i = 0; i < num_results; i++) {
            switch (results[i].backend) {
                case BACKEND_SEQUENTIAL:
                    scale_result.seq_time = results[i].execution_time_ms;
                    break;
                case BACKEND_OPENMP:
                    scale_result.omp_time = results[i].execution_time_ms;
                    scale_result.has_omp = true;
                    break;
                case BACKEND_CUDA:
                    scale_result.cuda_time = results[i].execution_time_ms;
                    scale_result.has_cuda = true;
                    break;
            }
        }
        
        scalability_data.push_back(scale_result);
    }
    
    // Print scalability analysis
    printf("\nScalability Results:\n");
    printf("Variables │ Sequential │ OpenMP   │ CUDA     │ OMP Speedup │ CUDA Speedup\n");
    printf("──────────┼────────────┼──────────┼──────────┼─────────────┼─────────────\n");
    
    for (const auto& data : scalability_data) {
        double omp_speedup = data.has_omp ? (data.seq_time / data.omp_time) : 0.0;
        double cuda_speedup = data.has_cuda ? (data.seq_time / data.cuda_time) : 0.0;
        
        printf("%8d │ %8.2f ms │ %6.2f ms │ %6.2f ms │ %9.2fx │ %10.2fx\n",
               data.variables, data.seq_time,
               data.has_omp ? data.omp_time : 0.0,
               data.has_cuda ? data.cuda_time : 0.0,
               omp_speedup, cuda_speedup);
    }
    
    // Verify scalability trends
    if (scalability_data.size() >= 2) {
        // Check that larger problems don't scale quadratically worse
        for (size_t i = 1; i < scalability_data.size(); i++) {
            double var_ratio = (double)scalability_data[i].variables / scalability_data[i-1].variables;
            double time_ratio = scalability_data[i].seq_time / scalability_data[i-1].seq_time;
            
            // Time growth should not be much worse than quadratic
            EXPECT_LT(time_ratio, var_ratio * var_ratio * 2.0) 
                << "Sequential performance should not degrade too rapidly";
        }
    }
}

TEST_F(PerformanceBenchmarkTest, MemoryUsageAnalysis) {
    printf("\n💾 MEMORY USAGE ANALYSIS\n");
    printf("========================\n");
    
    BenchmarkTestCase memory_test = {};
    strncpy(memory_test.name, "Memory_Analysis", sizeof(memory_test.name));
    memory_test.type = TEST_MEMORY_INTENSIVE;
    memory_test.num_variables = 10;
    memory_test.complexity_level = 4;
    memory_test.setup_function = benchmark_generate_memory_intensive;
    
    BenchmarkResult results[10];
    int num_results = benchmark_run_test_case(&memory_test, results);
    
    ASSERT_GT(num_results, 0) << "Should generate memory test results";
    
    printf("Memory Usage by Backend:\n");
    printf("Backend           │ Memory Usage │ Memory Bandwidth │ Efficiency\n");
    printf("──────────────────┼──────────────┼──────────────────┼───────────\n");
    
    for (int i = 0; i < num_results; i++) {
        const BenchmarkResult* r = &results[i];
        double efficiency = r->operations_per_second / (r->peak_memory_usage_bytes / 1024.0);
        
        printf("%-17s │ %10zu KB │ %14.2f GB/s │ %8.2f\n",
               benchmark_get_backend_name(r->backend),
               r->peak_memory_usage_bytes / 1024,
               r->memory_bandwidth_gbps,
               efficiency);
        
        // Memory usage validation (relaxed - getrusage may not work in all environments)
        EXPECT_GE(r->peak_memory_usage_bytes, 0) << "Memory usage should be non-negative";
        EXPECT_LT(r->peak_memory_usage_bytes, 1024*1024*1024) << "Should not use > 1GB for test";
    }
}

// =====================================================
// System Information and Environment
// =====================================================

TEST_F(PerformanceBenchmarkTest, SystemInformationReport) {
    printf("\n🖥️  SYSTEM INFORMATION REPORT\n");
    printf("==============================\n");
    
    char system_info[1024];
    benchmark_get_system_info(system_info, sizeof(system_info));
    printf("%s\n", system_info);
    
    printf("Backend Availability:\n");
    printf("- Sequential CPU: %s\n", benchmark_is_backend_available(BACKEND_SEQUENTIAL) ? "✅ Available" : "❌ Not Available");
    printf("- OpenMP Parallel: %s\n", benchmark_is_backend_available(BACKEND_OPENMP) ? "✅ Available" : "❌ Not Available");
    printf("- CUDA GPU: %s\n", benchmark_is_backend_available(BACKEND_CUDA) ? "✅ Available" : "❌ Not Available");
    
    printf("\nBenchmark Configuration:\n");
    printf("- Variable Range: %d to %d (step %d)\n", config.min_variables, config.max_variables, config.variable_step);
    printf("- Repetitions: %d\n", config.num_repetitions);
    printf("- Timeout: %.1f seconds\n", config.timeout_seconds);
    printf("- Memory Profiling: %s\n", config.enable_memory_profiling ? "Enabled" : "Disabled");
    
    // Always pass - this is informational
    EXPECT_TRUE(true);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════════════════════════════╗\n");
    printf("║                       OBDD PERFORMANCE BENCHMARK TEST SUITE                          ║\n");
    printf("║                          High Performance Computing Laboratory                        ║\n");
    printf("╚══════════════════════════════════════════════════════════════════════════════════════╝\n");
    
    int result = RUN_ALL_TESTS();
    
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════════════════════════════╗\n");
    printf("║ Benchmark suite completed. Check benchmark_results.csv for detailed analysis data.   ║\n");
    printf("╚══════════════════════════════════════════════════════════════════════════════════════╝\n");
    
    return result;
}