/**
 * @file test_large_scale_comparison.cpp
 * @brief Large-scale BDD performance comparison to find parallelization benefits
 * 
 * This test creates progressively larger BDD problems to identify the crossover
 * points where OpenMP and CUDA become advantageous over sequential execution.
 */

#include <gtest/gtest.h>
#include "core/obdd.hpp"
#include <chrono>
#include <iostream>
#include <vector>
#include <iomanip>
#include <omp.h>

class LargeScaleComparison : public ::testing::Test {
protected:
    struct BenchmarkResult {
        int variables;
        long sequential_ms;
        long openmp_ms;
        long cuda_ms;
        double openmp_speedup;
        double cuda_speedup;
        int bdd_nodes;
    };
    
    std::vector<BenchmarkResult> results;
    
    void SetUp() override {
        omp_set_num_threads(std::min(8, omp_get_max_threads()));
    }
    
    /**
     * Create a complex BDD structure that exercises parallelization
     */
    OBDD* create_complex_bdd(int variables) {
        std::vector<int> order(variables);
        for (int i = 0; i < variables; ++i) order[i] = i;
        
        OBDD* bdd = obdd_create(variables, order.data());
        
        // Create a complex structure with many levels and branches
        OBDDNode* current = obdd_constant(0);
        
        for (int v = variables - 1; v >= 0; --v) {
            // Create alternating patterns to force complex BDD structures
            if (v % 2 == 0) {
                current = obdd_node_create(v, current, obdd_constant(1));
            } else {
                current = obdd_node_create(v, obdd_constant(1), current);
            }
        }
        
        bdd->root = current;
        return bdd;
    }
    
    /**
     * Create BDD with exponential growth pattern
     */
    OBDD* create_exponential_bdd(int variables) {
        std::vector<int> order(variables);
        for (int i = 0; i < variables; ++i) order[i] = i;
        
        OBDD* bdd = obdd_create(variables, order.data());
        
        // Build a structure that grows exponentially with depth
        std::vector<OBDDNode*> level_nodes;
        level_nodes.push_back(obdd_constant(0));
        level_nodes.push_back(obdd_constant(1));
        
        for (int v = variables - 1; v >= 0; --v) {
            std::vector<OBDDNode*> next_level;
            
            for (size_t i = 0; i < level_nodes.size() && next_level.size() < 8; ++i) {
                for (size_t j = i + 1; j < level_nodes.size() && next_level.size() < 8; ++j) {
                    next_level.push_back(obdd_node_create(v, level_nodes[i], level_nodes[j]));
                }
            }
            
            if (next_level.empty()) {
                next_level.push_back(obdd_node_create(v, level_nodes[0], 
                    level_nodes.size() > 1 ? level_nodes[1] : level_nodes[0]));
            }
            
            level_nodes = next_level;
        }
        
        bdd->root = level_nodes.empty() ? obdd_constant(1) : level_nodes[0];
        return bdd;
    }
    
    /**
     * Count nodes in BDD for complexity measurement
     */
    int count_nodes(OBDDNode* node, std::set<OBDDNode*>& visited) {
        if (!node || visited.count(node)) return 0;
        visited.insert(node);
        
        if (is_leaf(node)) return 1;
        
        return 1 + count_nodes(node->lowChild, visited) + 
               count_nodes(node->highChild, visited);
    }
    
    int count_bdd_nodes(OBDD* bdd) {
        std::set<OBDDNode*> visited;
        return bdd ? count_nodes(bdd->root, visited) : 0;
    }
};

/**
 * Test the crossover point where parallel execution becomes beneficial
 */
TEST_F(LargeScaleComparison, FindParallelizationBenefits) {
    std::cout << "\n=== Large Scale Performance Comparison ===" << std::endl;
    std::cout << "Finding crossover points for parallelization benefits" << std::endl;
    
    std::cout << std::setw(8) << "Vars" 
              << std::setw(10) << "Nodes"
              << std::setw(12) << "Sequential" 
              << std::setw(12) << "OpenMP" 
              << std::setw(12) << "CUDA"
              << std::setw(12) << "OMP Speedup" 
              << std::setw(12) << "CUDA Speedup" << std::endl;
    std::cout << std::string(76, '-') << std::endl;
    
    // Test increasing problem sizes
    for (int vars = 8; vars <= 28; vars += 4) {
        BenchmarkResult result;
        result.variables = vars;
        
        // Create test BDDs
        OBDD* bdd1 = create_complex_bdd(vars);
        OBDD* bdd2 = create_exponential_bdd(vars);
        
        result.bdd_nodes = count_bdd_nodes(bdd1) + count_bdd_nodes(bdd2);
        
        // Sequential benchmark
        auto start_seq = std::chrono::high_resolution_clock::now();
        OBDDNode* seq_result = obdd_apply(bdd1, bdd2, OBDD_AND);
        auto end_seq = std::chrono::high_resolution_clock::now();
        result.sequential_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_seq - start_seq).count();
        
        // OpenMP benchmark
        auto start_omp = std::chrono::high_resolution_clock::now();
        OBDDNode* omp_result = obdd_parallel_apply_omp(bdd1, bdd2, OBDD_AND);
        auto end_omp = std::chrono::high_resolution_clock::now();
        result.openmp_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_omp - start_omp).count();
        
        // CUDA benchmark (if available)
        result.cuda_ms = 0;
        result.cuda_speedup = 0;
#ifdef OBDD_ENABLE_CUDA
        auto start_cuda = std::chrono::high_resolution_clock::now();
        OBDDNode* cuda_result = obdd_parallel_apply_cuda(bdd1, bdd2, OBDD_AND);
        auto end_cuda = std::chrono::high_resolution_clock::now();
        result.cuda_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_cuda - start_cuda).count();
        result.cuda_speedup = result.cuda_ms > 0 ? 
            (double)result.sequential_ms / result.cuda_ms : 0;
        
        ASSERT_NE(cuda_result, nullptr) << "CUDA result null for " << vars << " variables";
#endif
        
        // Calculate speedups
        result.openmp_speedup = result.openmp_ms > 0 ? 
            (double)result.sequential_ms / result.openmp_ms : 0;
        
        // Print results
        std::cout << std::setw(8) << vars
                  << std::setw(10) << result.bdd_nodes
                  << std::setw(12) << result.sequential_ms << "ms"
                  << std::setw(12) << result.openmp_ms << "ms"
                  << std::setw(12) << result.cuda_ms << "ms"
                  << std::setw(12) << std::fixed << std::setprecision(2) 
                  << result.openmp_speedup << "x"
                  << std::setw(12) << result.cuda_speedup << "x" << std::endl;
        
        results.push_back(result);
        
        // Verify correctness
        ASSERT_NE(seq_result, nullptr) << "Sequential result null for " << vars << " variables";
        ASSERT_NE(omp_result, nullptr) << "OpenMP result null for " << vars << " variables";
        
        // Cleanup
        obdd_destroy(bdd1);
        obdd_destroy(bdd2);
        
        // Stop if we're taking too long (>10 seconds per test)
        if (result.sequential_ms > 10000) {
            std::cout << "\n⚠️  Tests becoming too slow, stopping at " << vars << " variables" << std::endl;
            break;
        }
    }
    
    // Analyze results
    std::cout << "\n=== Analysis ===" << std::endl;
    
    // Find OpenMP crossover point
    for (const auto& result : results) {
        if (result.openmp_speedup > 1.0) {
            std::cout << "✅ OpenMP becomes beneficial at " << result.variables 
                      << " variables (speedup: " << result.openmp_speedup << "x)" << std::endl;
            break;
        }
    }
    
    // Find CUDA crossover point
    for (const auto& result : results) {
        if (result.cuda_speedup > 1.0) {
            std::cout << "✅ CUDA becomes beneficial at " << result.variables 
                      << " variables (speedup: " << result.cuda_speedup << "x)" << std::endl;
            break;
        }
    }
    
    // Find best performers
    if (!results.empty()) {
        auto max_omp = *std::max_element(results.begin(), results.end(),
            [](const BenchmarkResult& a, const BenchmarkResult& b) {
                return a.openmp_speedup < b.openmp_speedup;
            });
        
        auto max_cuda = *std::max_element(results.begin(), results.end(),
            [](const BenchmarkResult& a, const BenchmarkResult& b) {
                return a.cuda_speedup < b.cuda_speedup;
            });
            
        std::cout << "🏆 Best OpenMP speedup: " << max_omp.openmp_speedup 
                  << "x at " << max_omp.variables << " variables" << std::endl;
        std::cout << "🏆 Best CUDA speedup: " << max_cuda.cuda_speedup 
                  << "x at " << max_cuda.variables << " variables" << std::endl;
    }
}

/**
 * Test memory-intensive scenarios that should favor GPU
 */
TEST_F(LargeScaleComparison, MemoryIntensiveOperations) {
    std::cout << "\n=== Memory-Intensive Operations Test ===" << std::endl;
    
    // Create large BDDs that stress memory bandwidth
    for (int vars = 16; vars <= 24; vars += 4) {
        std::cout << "Testing " << vars << " variables..." << std::endl;
        
        OBDD* bdd1 = create_exponential_bdd(vars);
        OBDD* bdd2 = create_complex_bdd(vars);
        
        int total_nodes = count_bdd_nodes(bdd1) + count_bdd_nodes(bdd2);
        std::cout << "  Total BDD nodes: " << total_nodes << std::endl;
        
        // Sequential
        auto start = std::chrono::high_resolution_clock::now();
        OBDDNode* seq_result = obdd_apply(bdd1, bdd2, OBDD_XOR);
        auto end = std::chrono::high_resolution_clock::now();
        long seq_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        
        // OpenMP
        start = std::chrono::high_resolution_clock::now();
        OBDDNode* omp_result = obdd_parallel_apply_omp(bdd1, bdd2, OBDD_XOR);
        end = std::chrono::high_resolution_clock::now();
        long omp_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        
        double speedup = omp_time > 0 ? (double)seq_time / omp_time : 0;
        
        std::cout << "  Sequential: " << seq_time << "ms, OpenMP: " << omp_time 
                  << "ms, Speedup: " << speedup << "x" << std::endl;
        
        ASSERT_NE(seq_result, nullptr);
        ASSERT_NE(omp_result, nullptr);
        
        obdd_destroy(bdd1);
        obdd_destroy(bdd2);
        
        if (seq_time > 15000) {  // Stop if >15 seconds
            std::cout << "  ⚠️  Test taking too long, stopping" << std::endl;
            break;
        }
    }
}