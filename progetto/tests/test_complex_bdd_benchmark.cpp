/**
 * @file test_complex_bdd_benchmark.cpp
 * @brief Complex BDD benchmark to stress test parallelization
 */

#include <gtest/gtest.h>
#include "core/obdd.hpp"
#include <chrono>
#include <iostream>
#include <vector>
#include <iomanip>
#include <omp.h>

class ComplexBDDBenchmark : public ::testing::Test {
protected:
    void SetUp() override {
        omp_set_num_threads(std::min(8, omp_get_max_threads()));
    }
    
    /**
     * Create a BDD that represents a complex Boolean function
     * This creates many recursive apply operations
     */
    OBDD* create_complex_function(int variables) {
        std::vector<int> order(variables);
        for (int i = 0; i < variables; ++i) order[i] = i;
        
        OBDD* result = obdd_create(variables, order.data());
        OBDD* temp1 = obdd_create(variables, order.data());
        OBDD* temp2 = obdd_create(variables, order.data());
        
        // Create variable nodes
        std::vector<OBDDNode*> var_nodes(variables);
        for (int i = 0; i < variables; ++i) {
            var_nodes[i] = obdd_node_create(i, obdd_constant(0), obdd_constant(1));
        }
        
        // Build complex expression: (x0 AND x1) OR (x2 AND x3) OR ... with many levels
        OBDDNode* current = obdd_constant(0);
        
        for (int i = 0; i < variables - 1; i += 2) {
            // Create (xi AND x(i+1))
            temp1->root = var_nodes[i];
            temp2->root = var_nodes[i + 1];
            
            OBDDNode* and_result = obdd_apply(temp1, temp2, OBDD_AND);
            
            // OR with previous result
            if (current == obdd_constant(0)) {
                current = and_result;
            } else {
                temp1->root = current;
                temp2->root = and_result;
                current = obdd_apply(temp1, temp2, OBDD_OR);
            }
        }
        
        result->root = current;
        
        obdd_destroy(temp1);
        obdd_destroy(temp2);
        
        return result;
    }
    
    /**
     * Perform many operations on the BDD to create computational load
     */
    long benchmark_operations(OBDD* bdd1, OBDD* bdd2, bool use_openmp) {
        auto start = std::chrono::high_resolution_clock::now();
        
        // Perform multiple complex operations
        std::vector<OBDDNode*> results;
        
        for (int iter = 0; iter < 100; ++iter) {  // Many iterations to accumulate time
            OBDDNode* and_result, *or_result, *xor_result;
            
            if (use_openmp) {
                and_result = obdd_parallel_apply_omp(bdd1, bdd2, OBDD_AND);
                or_result = obdd_parallel_apply_omp(bdd1, bdd2, OBDD_OR);
                xor_result = obdd_parallel_apply_omp(bdd1, bdd2, OBDD_XOR);
            } else {
                and_result = obdd_apply(bdd1, bdd2, OBDD_AND);
                or_result = obdd_apply(bdd1, bdd2, OBDD_OR);
                xor_result = obdd_apply(bdd1, bdd2, OBDD_XOR);
            }
            
            // Store results to prevent optimization elimination
            results.push_back(and_result);
            results.push_back(or_result);
            results.push_back(xor_result);
            
            // Clear some results to avoid memory explosion
            if (iter % 20 == 19) {
                results.clear();
            }
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    }
};

TEST_F(ComplexBDDBenchmark, ComparativePerformance) {
    std::cout << "\n=== Complex BDD Performance Benchmark ===" << std::endl;
    std::cout << "Testing with intensive operations to measure parallelization benefits" << std::endl;
    
    std::cout << std::setw(8) << "Vars"
              << std::setw(15) << "Sequential"
              << std::setw(15) << "OpenMP"
              << std::setw(15) << "Speedup" << std::endl;
    std::cout << std::string(53, '-') << std::endl;
    
    for (int vars = 6; vars <= 20; vars += 2) {
        std::cout << "Creating BDDs with " << vars << " variables..." << std::endl;
        
        OBDD* bdd1 = create_complex_function(vars);
        OBDD* bdd2 = create_complex_function(vars);
        
        // Sequential benchmark
        std::cout << "  Running sequential benchmark..." << std::endl;
        long seq_time = benchmark_operations(bdd1, bdd2, false);
        
        // OpenMP benchmark
        std::cout << "  Running OpenMP benchmark..." << std::endl;
        long omp_time = benchmark_operations(bdd1, bdd2, true);
        
        double speedup = omp_time > 0 ? (double)seq_time / omp_time : 0;
        
        std::cout << std::setw(8) << vars
                  << std::setw(15) << seq_time << "ms"
                  << std::setw(15) << omp_time << "ms"
                  << std::setw(15) << std::fixed << std::setprecision(2) 
                  << speedup << "x" << std::endl;
        
        // Validate results exist
        ASSERT_NE(bdd1->root, nullptr);
        ASSERT_NE(bdd2->root, nullptr);
        
        obdd_destroy(bdd1);
        obdd_destroy(bdd2);
        
        // If we found speedup > 1, report it
        if (speedup > 1.0) {
            std::cout << "✅ OpenMP shows speedup at " << vars << " variables!" << std::endl;
        }
        
        // Stop if tests are too slow (>30 seconds per test)
        if (seq_time > 30000) {
            std::cout << "⚠️  Tests becoming too slow, stopping." << std::endl;
            break;
        }
    }
}

/**
 * Test recursive depth scenarios that should benefit from parallelization
 */
TEST_F(ComplexBDDBenchmark, RecursiveDepthTest) {
    std::cout << "\n=== Recursive Depth Performance Test ===" << std::endl;
    
    for (int depth = 8; depth <= 16; depth += 2) {
        std::cout << "Testing recursive depth " << depth << "..." << std::endl;
        
        std::vector<int> order(depth);
        for (int i = 0; i < depth; ++i) order[i] = i;
        
        // Create deeply nested BDD structure
        OBDD* bdd1 = obdd_create(depth, order.data());
        OBDD* bdd2 = obdd_create(depth, order.data());
        
        // Build nested structure
        OBDDNode* nested1 = obdd_constant(1);
        OBDDNode* nested2 = obdd_constant(0);
        
        for (int level = depth - 1; level >= 0; --level) {
            nested1 = obdd_node_create(level, 
                level % 2 == 0 ? obdd_constant(0) : nested1,
                level % 2 == 0 ? nested1 : obdd_constant(1));
                
            nested2 = obdd_node_create(level,
                level % 3 == 0 ? nested2 : obdd_constant(1), 
                level % 3 == 0 ? obdd_constant(0) : nested2);
        }
        
        bdd1->root = nested1;
        bdd2->root = nested2;
        
        // Time the operations
        auto seq_start = std::chrono::high_resolution_clock::now();
        
        // Multiple operations to get measurable time
        for (int i = 0; i < 50; ++i) {
            OBDDNode* result = obdd_apply(bdd1, bdd2, OBDD_AND);
            (void)result; // Prevent optimization
        }
        
        auto seq_end = std::chrono::high_resolution_clock::now();
        long seq_time = std::chrono::duration_cast<std::chrono::milliseconds>(seq_end - seq_start).count();
        
        auto omp_start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < 50; ++i) {
            OBDDNode* result = obdd_parallel_apply_omp(bdd1, bdd2, OBDD_AND);
            (void)result;
        }
        
        auto omp_end = std::chrono::high_resolution_clock::now();
        long omp_time = std::chrono::duration_cast<std::chrono::milliseconds>(omp_end - omp_start).count();
        
        double speedup = omp_time > 0 ? (double)seq_time / omp_time : 0;
        
        std::cout << "  Depth " << depth 
                  << ": Sequential " << seq_time << "ms"
                  << ", OpenMP " << omp_time << "ms"
                  << ", Speedup " << speedup << "x" << std::endl;
        
        if (speedup > 1.0) {
            std::cout << "  ✅ OpenMP beneficial at depth " << depth << std::endl;
        }
        
        obdd_destroy(bdd1);
        obdd_destroy(bdd2);
        
        if (seq_time > 10000) {
            std::cout << "  ⚠️  Test taking too long, stopping" << std::endl;
            break;
        }
    }
}