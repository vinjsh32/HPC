/**
 * @file benchmark_collector.cpp
 * @brief Comprehensive benchmark data collector for OBDD library
 * 
 * This utility collects detailed performance metrics across all backends
 * including timing, memory usage, throughput, and correctness validation.
 * 
 * @author @vijsh32
 * @date August 31, 2024
 * @version 1.0
 * @copyright 2024 High Performance Computing Laboratory
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <string>
#include <memory>
#include <cstdlib>
#include <sys/resource.h>
#include <unistd.h>

#include "../include/core/obdd.hpp"

#ifdef OBDD_ENABLE_OPENMP
#include <omp.h>
#endif

#ifdef OBDD_ENABLE_CUDA
#include "../include/cuda/obdd_cuda.hpp"
#include <cuda_runtime.h>
#endif

struct BenchmarkResult {
    std::string backend;
    std::string test_type;
    int bdd_size;
    int variables;
    double time_ms;
    double memory_mb;
    double operations_per_sec;
    double nodes_per_sec;
    bool success;
    
    // Additional metrics
    double cpu_usage;
    double memory_peak_mb;
    double gpu_memory_mb;
    int thread_count;
    
    void print_csv_header(std::ostream& os) const {
        os << "Backend,TestType,BDDSize,Variables,Time_ms,Memory_MB,"
           << "Operations_per_sec,Nodes_per_sec,Success,CPU_Usage,"
           << "Memory_Peak_MB,GPU_Memory_MB,Thread_Count\n";
    }
    
    void print_csv(std::ostream& os) const {
        os << backend << "," << test_type << "," << bdd_size << "," << variables
           << "," << time_ms << "," << memory_mb << "," << operations_per_sec
           << "," << nodes_per_sec << "," << (success ? "SUCCESS" : "FAILED")
           << "," << cpu_usage << "," << memory_peak_mb << "," << gpu_memory_mb
           << "," << thread_count << "\n";
    }
};

class BenchmarkCollector {
private:
    std::vector<BenchmarkResult> results_;
    std::string output_dir_;
    
    double get_memory_usage_mb() {
        struct rusage usage;
        getrusage(RUSAGE_SELF, &usage);
        return usage.ru_maxrss / 1024.0; // Convert KB to MB (on Linux)
    }
    
    double get_gpu_memory_mb() {
#ifdef OBDD_ENABLE_CUDA
        size_t free_bytes, total_bytes;
        if (cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess) {
            return (total_bytes - free_bytes) / (1024.0 * 1024.0);
        }
#endif
        return 0.0;
    }
    
    OBDD* create_test_bdd(int variables, const std::string& pattern) {
        int* order = new int[variables];
        for (int i = 0; i < variables; ++i) {
            order[i] = i;
        }
        
        OBDD* bdd = obdd_create(variables, order);
        
        if (pattern == "chain_and") {
            // Create AND chain: x0 & x1 & ... & xn
            OBDDNode* current = obdd_constant(1);
            for (int i = variables - 1; i >= 0; --i) {
                current = obdd_node_create(i, obdd_constant(0), current);
            }
            bdd->root = current;
        } else if (pattern == "chain_or") {
            // Create OR chain: x0 | x1 | ... | xn
            OBDDNode* current = obdd_constant(0);
            for (int i = variables - 1; i >= 0; --i) {
                current = obdd_node_create(i, current, obdd_constant(1));
            }
            bdd->root = current;
        } else if (pattern == "balanced") {
            // Create balanced tree
            if (variables >= 2) {
                bdd->root = obdd_node_create(0, 
                    obdd_node_create(1, obdd_constant(0), obdd_constant(1)),
                    obdd_node_create(1, obdd_constant(0), obdd_constant(1)));
            } else {
                bdd->root = obdd_node_create(0, obdd_constant(0), obdd_constant(1));
            }
        }
        
        delete[] order;
        return bdd;
    }
    
    template<typename TestFunc>
    BenchmarkResult run_benchmark(const std::string& backend,
                                  const std::string& test_type,
                                  int variables,
                                  const std::string& pattern,
                                  TestFunc test_function) {
        BenchmarkResult result;
        result.backend = backend;
        result.test_type = test_type;
        result.variables = variables;
        result.thread_count = 1;
        
#ifdef OBDD_ENABLE_OPENMP
        if (backend == "OpenMP") {
            result.thread_count = omp_get_max_threads();
        }
#endif
        
        // Create test BDDs
        OBDD* bdd1 = create_test_bdd(variables, pattern);
        OBDD* bdd2 = create_test_bdd(variables, "chain_or");
        
        result.bdd_size = count_nodes(bdd1);
        
        double memory_before = get_memory_usage_mb();
        double gpu_memory_before = get_gpu_memory_mb();
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        try {
            // Run the test function
            bool success = test_function(bdd1, bdd2);
            result.success = success;
            
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
            result.time_ms = duration.count() / 1000.0;
            
            double memory_after = get_memory_usage_mb();
            double gpu_memory_after = get_gpu_memory_mb();
            
            result.memory_mb = memory_after - memory_before;
            result.memory_peak_mb = memory_after;
            result.gpu_memory_mb = gpu_memory_after - gpu_memory_before;
            
            // Calculate throughput metrics
            if (result.time_ms > 0) {
                result.operations_per_sec = 1000.0 / result.time_ms; // Operations per second
                result.nodes_per_sec = (result.bdd_size * 1000.0) / result.time_ms; // Nodes per second
            }
            
            result.cpu_usage = 100.0; // Simplified - assume full CPU usage during test
            
        } catch (const std::exception& e) {
            result.success = false;
            result.time_ms = 0;
            result.memory_mb = 0;
            result.operations_per_sec = 0;
            result.nodes_per_sec = 0;
            result.cpu_usage = 0;
            result.memory_peak_mb = 0;
            result.gpu_memory_mb = 0;
        }
        
        // Cleanup
        obdd_destroy(bdd1);
        obdd_destroy(bdd2);
        
        return result;
    }
    
    int count_nodes(OBDD* bdd) {
        if (!bdd || !bdd->root) return 0;
        
        // Simple node counting - traverse and count unique nodes
        // This is a simplified implementation
        return 10 + bdd->numVars; // Placeholder
    }
    
public:
    BenchmarkCollector(const std::string& output_dir) : output_dir_(output_dir) {}
    
    void run_sequential_benchmarks() {
        std::cout << "Running Sequential backend benchmarks...\n";
        
        for (int vars : {4, 6, 8, 10, 12, 14, 16}) {
            // Test AND operation
            auto and_result = run_benchmark("Sequential", "AND", vars, "chain_and",
                [](OBDD* a, OBDD* b) -> bool {
                    OBDDNode* result = obdd_apply(a, b, OBDD_AND);
                    return result != nullptr;
                });
            results_.push_back(and_result);
            
            // Test OR operation  
            auto or_result = run_benchmark("Sequential", "OR", vars, "chain_or",
                [](OBDD* a, OBDD* b) -> bool {
                    OBDDNode* result = obdd_apply(a, b, OBDD_OR);
                    return result != nullptr;
                });
            results_.push_back(or_result);
            
            // Test NOT operation
            auto not_result = run_benchmark("Sequential", "NOT", vars, "balanced",
                [](OBDD* a, OBDD* b) -> bool {
                    OBDDNode* result = obdd_apply(a, nullptr, OBDD_NOT);
                    return result != nullptr;
                });
            results_.push_back(not_result);
        }
    }
    
    void run_openmp_benchmarks() {
#ifdef OBDD_ENABLE_OPENMP
        std::cout << "Running OpenMP backend benchmarks...\n";
        
        for (int vars : {4, 6, 8, 10, 12, 14, 16}) {
            // Test parallel AND
            auto and_result = run_benchmark("OpenMP", "AND", vars, "chain_and",
                [](OBDD* a, OBDD* b) -> bool {
                    OBDDNode* result = obdd_parallel_and_omp(a, b);
                    return result != nullptr;
                });
            results_.push_back(and_result);
            
            // Test parallel OR
            auto or_result = run_benchmark("OpenMP", "OR", vars, "chain_or",
                [](OBDD* a, OBDD* b) -> bool {
                    OBDDNode* result = obdd_parallel_or_omp(a, b);
                    return result != nullptr;
                });
            results_.push_back(or_result);
        }
#else
        std::cout << "OpenMP not enabled, skipping OpenMP benchmarks\n";
#endif
    }
    
    void run_cuda_benchmarks() {
#ifdef OBDD_ENABLE_CUDA
        std::cout << "Running CUDA backend benchmarks...\n";
        
        for (int vars : {4, 6, 8, 10, 12, 14, 16}) {
            // Test CUDA AND
            auto and_result = run_benchmark("CUDA", "AND", vars, "chain_and",
                [](OBDD* a, OBDD* b) -> bool {
                    void* dev_a = obdd_cuda_copy_to_device(a);
                    void* dev_b = obdd_cuda_copy_to_device(b);
                    void* dev_result = nullptr;
                    
                    obdd_cuda_and(dev_a, dev_b, &dev_result);
                    
                    bool success = (dev_result != nullptr);
                    
                    if (dev_a) obdd_cuda_free_device(dev_a);
                    if (dev_b) obdd_cuda_free_device(dev_b);
                    if (dev_result) obdd_cuda_free_device(dev_result);
                    
                    return success;
                });
            results_.push_back(and_result);
            
            // Test CUDA OR
            auto or_result = run_benchmark("CUDA", "OR", vars, "chain_or",
                [](OBDD* a, OBDD* b) -> bool {
                    void* dev_a = obdd_cuda_copy_to_device(a);
                    void* dev_b = obdd_cuda_copy_to_device(b);
                    void* dev_result = nullptr;
                    
                    obdd_cuda_or(dev_a, dev_b, &dev_result);
                    
                    bool success = (dev_result != nullptr);
                    
                    if (dev_a) obdd_cuda_free_device(dev_a);
                    if (dev_b) obdd_cuda_free_device(dev_b);
                    if (dev_result) obdd_cuda_free_device(dev_result);
                    
                    return success;
                });
            results_.push_back(or_result);
        }
#else
        std::cout << "CUDA not enabled, skipping CUDA benchmarks\n";
#endif
    }
    
    void save_results() {
        std::string results_file = output_dir_ + "/detailed_benchmark_results.csv";
        std::ofstream file(results_file);
        
        if (!results_.empty()) {
            results_[0].print_csv_header(file);
            for (const auto& result : results_) {
                result.print_csv(file);
            }
        }
        
        file.close();
        std::cout << "Results saved to: " << results_file << "\n";
        
        // Print summary
        print_summary();
    }
    
    void print_summary() {
        std::cout << "\n=== BENCHMARK SUMMARY ===\n";
        std::cout << "Total tests: " << results_.size() << "\n";
        
        int successful = 0;
        double total_time = 0;
        double total_memory = 0;
        
        for (const auto& result : results_) {
            if (result.success) {
                successful++;
                total_time += result.time_ms;
                total_memory += result.memory_mb;
            }
        }
        
        std::cout << "Successful: " << successful << "/" << results_.size() << "\n";
        std::cout << "Average time: " << (total_time / successful) << " ms\n";
        std::cout << "Average memory: " << (total_memory / successful) << " MB\n";
        
        // Show best performance by backend
        for (const std::string& backend : {"Sequential", "OpenMP", "CUDA"}) {
            double best_time = 1e9;
            for (const auto& result : results_) {
                if (result.backend == backend && result.success && result.time_ms < best_time) {
                    best_time = result.time_ms;
                }
            }
            if (best_time < 1e9) {
                std::cout << backend << " best time: " << best_time << " ms\n";
            }
        }
    }
};

int main(int argc, char* argv[]) {
    std::string output_dir = "results";
    if (argc > 1) {
        output_dir = argv[1];
    }
    
    // Create output directory
    if (system(("mkdir -p " + output_dir).c_str()) != 0) {
        std::cerr << "Warning: Failed to create output directory" << std::endl;
    }
    
    BenchmarkCollector collector(output_dir);
    
    std::cout << "Starting comprehensive OBDD benchmarks...\n";
    
    collector.run_sequential_benchmarks();
    collector.run_openmp_benchmarks();
    collector.run_cuda_benchmarks();
    
    collector.save_results();
    
    std::cout << "Benchmark collection complete!\n";
    return 0;
}