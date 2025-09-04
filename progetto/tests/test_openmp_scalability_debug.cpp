/**
 * @file test_openmp_scalability_debug.cpp
 * @brief Debug test to identify OpenMP timeout issues in scalability tests
 * 
 * This test isolates the specific conditions that cause OpenMP timeouts
 * in scalability benchmarks to identify and fix the root cause.
 */

#include <gtest/gtest.h>
#include "core/obdd.hpp"
#include <chrono>
#include <iostream>
#include <omp.h>

class OpenMPScalabilityDebug : public ::testing::Test {
protected:
    void SetUp() override {
        // Set conservative thread count to avoid resource exhaustion
        original_threads = omp_get_max_threads();
        omp_set_num_threads(std::min(4, original_threads));
    }
    
    void TearDown() override {
        // Restore original thread count
        omp_set_num_threads(original_threads);
    }
    
private:
    int original_threads;
};

/**
 * Test simple OpenMP operations that should not timeout
 */
TEST_F(OpenMPScalabilityDebug, SimpleOperations) {
    std::cout << "Testing simple OpenMP operations..." << std::endl;
    
    int order[] = {0, 1, 2, 3, 4, 5, 6, 7};
    OBDD* bdd1 = obdd_create(8, order);
    OBDD* bdd2 = obdd_create(8, order);
    
    // Create simple non-trivial BDDs
    bdd1->root = obdd_node_create(0, obdd_constant(0), obdd_constant(1));
    bdd2->root = obdd_node_create(0, obdd_constant(1), obdd_constant(0));
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Test basic operations
    OBDDNode* result_and = obdd_parallel_and_omp(bdd1, bdd2);
    OBDDNode* result_or = obdd_parallel_or_omp(bdd1, bdd2);
    OBDDNode* result_xor = obdd_parallel_xor_omp(bdd1, bdd2);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "Simple operations completed in " << duration.count() << "ms" << std::endl;
    
    ASSERT_NE(result_and, nullptr);
    ASSERT_NE(result_or, nullptr);
    ASSERT_NE(result_xor, nullptr);
    
    obdd_destroy(bdd1);
    obdd_destroy(bdd2);
}

/**
 * Test that reproduces the exact scalability test conditions
 */
TEST_F(OpenMPScalabilityDebug, ScalabilityTest8Variables) {
    std::cout << "Testing OpenMP with 8 variables (reproducing benchmark conditions)..." << std::endl;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    int order[] = {0, 1, 2, 3, 4, 5, 6, 7};
    OBDD* bdd1 = obdd_create(8, order);
    OBDD* bdd2 = obdd_create(8, order);
    
    // Create more complex BDD structures like in scalability tests
    OBDDNode* node_4 = obdd_node_create(4, obdd_constant(0), obdd_constant(1));
    OBDDNode* node_3 = obdd_node_create(3, obdd_constant(1), node_4);
    OBDDNode* node_2 = obdd_node_create(2, node_4, node_3);
    OBDDNode* node_1 = obdd_node_create(1, node_3, node_2);
    bdd1->root = obdd_node_create(0, node_2, node_1);
    
    OBDDNode* node2_4 = obdd_node_create(4, obdd_constant(1), obdd_constant(0));
    OBDDNode* node2_3 = obdd_node_create(3, node2_4, obdd_constant(0));
    OBDDNode* node2_2 = obdd_node_create(2, node2_3, node2_4);
    OBDDNode* node2_1 = obdd_node_create(1, node2_2, node2_3);
    bdd2->root = obdd_node_create(0, node2_1, node2_2);
    
    std::cout << "BDDs created, starting parallel operations..." << std::endl;
    
    // This is where timeout might occur
    OBDDNode* result = obdd_parallel_and_omp(bdd1, bdd2);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "8-variable scalability test completed in " << duration.count() << "ms" << std::endl;
    
    ASSERT_NE(result, nullptr);
    
    obdd_destroy(bdd1);
    obdd_destroy(bdd2);
}

/**
 * Test with increasing complexity to find the breakpoint
 */
TEST_F(OpenMPScalabilityDebug, ProgressiveComplexity) {
    for (int vars = 6; vars <= 12; vars += 2) {
        std::cout << "Testing OpenMP with " << vars << " variables..." << std::endl;
        
        auto start = std::chrono::high_resolution_clock::now();
        
        std::vector<int> order(vars);
        for (int i = 0; i < vars; ++i) order[i] = i;
        
        OBDD* bdd1 = obdd_create(vars, order.data());
        OBDD* bdd2 = obdd_create(vars, order.data());
        
        // Create progressively complex structures
        OBDDNode* current1 = obdd_constant(0);
        OBDDNode* current2 = obdd_constant(1);
        
        for (int v = vars - 1; v >= 0; --v) {
            OBDDNode* new1 = obdd_node_create(v, current1, obdd_constant(1));
            OBDDNode* new2 = obdd_node_create(v, obdd_constant(0), current2);
            current1 = new1;
            current2 = new2;
        }
        
        bdd1->root = current1;
        bdd2->root = current2;
        
        // Test the operation that might timeout
        OBDDNode* result = obdd_parallel_apply_omp(bdd1, bdd2, OBDD_AND);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        std::cout << vars << " variables completed in " << duration.count() << "ms" << std::endl;
        
        ASSERT_NE(result, nullptr) << "Failed at " << vars << " variables";
        
        obdd_destroy(bdd1);
        obdd_destroy(bdd2);
        
        // Early exit if we're taking too long
        if (duration.count() > 5000) {
            std::cout << "⚠️  Test taking too long at " << vars << " variables, stopping" << std::endl;
            break;
        }
    }
}

/**
 * Test the specific cache and threading issues that might cause deadlocks
 */
TEST_F(OpenMPScalabilityDebug, ThreadingAndCacheTest) {
    std::cout << "Testing threading and cache behavior..." << std::endl;
    
    // Test with different thread counts to identify threading issues
    for (int num_threads = 1; num_threads <= 4; ++num_threads) {
        omp_set_num_threads(num_threads);
        std::cout << "Testing with " << num_threads << " threads..." << std::endl;
        
        int order[] = {0, 1, 2, 3, 4, 5, 6, 7};
        OBDD* bdd1 = obdd_create(8, order);
        OBDD* bdd2 = obdd_create(8, order);
        
        // Create identical structures to stress cache
        bdd1->root = obdd_node_create(0, 
            obdd_node_create(1, obdd_constant(0), obdd_constant(1)),
            obdd_node_create(2, obdd_constant(1), obdd_constant(0))
        );
        bdd2->root = bdd1->root; // Same structure to test cache behavior
        
        auto start = std::chrono::high_resolution_clock::now();
        
        // Multiple operations to stress threading
        for (int i = 0; i < 5; ++i) {
            OBDDNode* result = obdd_parallel_apply_omp(bdd1, bdd2, OBDD_AND);
            ASSERT_NE(result, nullptr) << "Failed at thread count " << num_threads << ", iteration " << i;
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        std::cout << num_threads << " threads: " << duration.count() << "ms" << std::endl;
        
        obdd_destroy(bdd1);
        obdd_destroy(bdd2);
    }
}