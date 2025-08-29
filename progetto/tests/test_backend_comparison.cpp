/**
 * @file test_backend_comparison.cpp
 * @brief Cross-backend comparison and consistency tests
 */

#include "core/obdd.hpp"
#include "advanced/obdd_advanced_math.hpp"
#include <gtest/gtest.h>
#include <vector>
#include <chrono>

#ifdef OBDD_ENABLE_CUDA
#include "cuda/obdd_cuda.hpp"
#include "cuda/obdd_cuda_types.cuh"
#include <cuda_runtime.h>
#endif

class BackendComparisonTest : public ::testing::Test {
protected:
    static int eval_bdd_host(const OBDD* bdd, const int* assignment) {
        return obdd_evaluate(bdd, assignment);
    }
    
#ifdef OBDD_ENABLE_CUDA
    static int eval_bdd_cuda(void* dHandle, const int* assignment) {
        DeviceOBDD dev{};
        cudaMemcpy(&dev, dHandle, sizeof(DeviceOBDD), cudaMemcpyDeviceToHost);
        std::vector<NodeGPU> nodes(dev.size);
        cudaMemcpy(nodes.data(), dev.nodes, sizeof(NodeGPU) * dev.size,
                   cudaMemcpyDeviceToHost);
        
        if (dev.size <= 2) {
            if (dev.size == 2) return nodes[0].low;
            return nodes[dev.size - 1].low;
        }
        
        int rootIdx = -1;
        for (int i = dev.size - 1; i >= 2; --i) {
            if (nodes[i].var >= 0 && nodes[i].low != -1 && nodes[i].high != -1) {
                rootIdx = i;
                break;
            }
        }
        
        if (rootIdx == -1) {
            for (int i = dev.size - 1; i >= 2; --i) {
                if (nodes[i].var == -1) return nodes[i].low;
            }
            return nodes[dev.size > 1 ? dev.size - 1 : 0].low;
        }
        
        int idx = rootIdx;
        while (idx >= 0 && idx < dev.size && nodes[idx].var >= 0) {
            int var = nodes[idx].var;
            if (var >= dev.nVars) return 0;
            
            int next_idx = assignment[var] ? nodes[idx].high : nodes[idx].low;
            if (next_idx < 0 || next_idx >= dev.size) return 0;
            idx = next_idx;
        }
        
        return (idx >= 0 && idx < dev.size) ? nodes[idx].low : 0;
    }
#endif
};

TEST_F(BackendComparisonTest, BasicOperationConsistency) {
    int order[4] = {0,1,2,3};
    
    // Create test BDDs
    OBDD* bdd1 = obdd_create(4, order);
    OBDD* bdd2 = obdd_create(4, order);
    
    bdd1->root = obdd_node_create(0, obdd_constant(0), obdd_constant(1)); // x0
    bdd2->root = obdd_node_create(1, obdd_constant(0), obdd_constant(1)); // x1
    
    // Sequential operations
    OBDDNode* seq_and = obdd_apply(bdd1, bdd2, OBDD_AND);
    OBDDNode* seq_or = obdd_apply(bdd1, bdd2, OBDD_OR);
    OBDDNode* seq_xor = obdd_apply(bdd1, bdd2, OBDD_XOR);
    OBDDNode* seq_not = obdd_apply(bdd1, nullptr, OBDD_NOT);
    
    OBDD* seq_and_bdd = obdd_create(4, order);
    OBDD* seq_or_bdd = obdd_create(4, order);
    OBDD* seq_xor_bdd = obdd_create(4, order);
    OBDD* seq_not_bdd = obdd_create(4, order);
    
    seq_and_bdd->root = seq_and;
    seq_or_bdd->root = seq_or;
    seq_xor_bdd->root = seq_xor;
    seq_not_bdd->root = seq_not;
    
#ifdef OBDD_ENABLE_OPENMP
    // OpenMP operations
    OBDDNode* omp_and = obdd_parallel_and_omp(bdd1, bdd2);
    OBDDNode* omp_or = obdd_parallel_or_omp(bdd1, bdd2);
    OBDDNode* omp_not = obdd_parallel_not_omp(bdd1);
    
    OBDD* omp_and_bdd = obdd_create(4, order);
    OBDD* omp_or_bdd = obdd_create(4, order);
    OBDD* omp_not_bdd = obdd_create(4, order);
    
    omp_and_bdd->root = omp_and;
    omp_or_bdd->root = omp_or;
    omp_not_bdd->root = omp_not;
#endif

#ifdef OBDD_ENABLE_CUDA
    // CUDA operations
    void* cuda_bdd1 = obdd_cuda_copy_to_device(bdd1);
    void* cuda_bdd2 = obdd_cuda_copy_to_device(bdd2);
    
    void *cuda_and = nullptr, *cuda_or = nullptr, *cuda_xor = nullptr, *cuda_not = nullptr;
    obdd_cuda_and(cuda_bdd1, cuda_bdd2, &cuda_and);
    obdd_cuda_or(cuda_bdd1, cuda_bdd2, &cuda_or);
    obdd_cuda_xor(cuda_bdd1, cuda_bdd2, &cuda_xor);
    obdd_cuda_not(cuda_bdd1, &cuda_not);
#endif
    
    // Test all combinations of inputs
    for (int i = 0; i < 16; i++) {
        int assignment[4] = {
            (i >> 0) & 1,
            (i >> 1) & 1, 
            (i >> 2) & 1,
            (i >> 3) & 1
        };
        
        // Sequential results
        int seq_and_result = eval_bdd_host(seq_and_bdd, assignment);
        int seq_or_result = eval_bdd_host(seq_or_bdd, assignment);
        int seq_xor_result = eval_bdd_host(seq_xor_bdd, assignment);
        int seq_not_result = eval_bdd_host(seq_not_bdd, assignment);
        
#ifdef OBDD_ENABLE_OPENMP
        // OpenMP results should match sequential
        int omp_and_result = eval_bdd_host(omp_and_bdd, assignment);
        int omp_or_result = eval_bdd_host(omp_or_bdd, assignment);
        int omp_not_result = eval_bdd_host(omp_not_bdd, assignment);
        
        EXPECT_EQ(seq_and_result, omp_and_result) 
            << "AND mismatch at assignment " << i;
        EXPECT_EQ(seq_or_result, omp_or_result)
            << "OR mismatch at assignment " << i;
        EXPECT_EQ(seq_not_result, omp_not_result)
            << "NOT mismatch at assignment " << i;
#endif

#ifdef OBDD_ENABLE_CUDA
        // CUDA results 
        int cuda_and_result = eval_bdd_cuda(cuda_and, assignment);
        int cuda_or_result = eval_bdd_cuda(cuda_or, assignment);
        int cuda_xor_result = eval_bdd_cuda(cuda_xor, assignment);
        int cuda_not_result = eval_bdd_cuda(cuda_not, assignment);
        
        // Expected results based on truth tables
        int expected_and = assignment[0] & assignment[1];
        int expected_or = assignment[0] | assignment[1];
        int expected_xor = assignment[0] ^ assignment[1];
        int expected_not = !assignment[0];
        
        EXPECT_EQ(expected_and, cuda_and_result)
            << "CUDA AND mismatch at assignment " << i;
        EXPECT_EQ(expected_or, cuda_or_result)
            << "CUDA OR mismatch at assignment " << i;
        EXPECT_EQ(expected_xor, cuda_xor_result)
            << "CUDA XOR mismatch at assignment " << i;
        EXPECT_EQ(expected_not, cuda_not_result)
            << "CUDA NOT mismatch at assignment " << i;
#endif
    }
    
    // Cleanup
    obdd_destroy(bdd1);
    obdd_destroy(bdd2);
    obdd_destroy(seq_and_bdd);
    obdd_destroy(seq_or_bdd);
    obdd_destroy(seq_xor_bdd);
    obdd_destroy(seq_not_bdd);
    
#ifdef OBDD_ENABLE_OPENMP
    obdd_destroy(omp_and_bdd);
    obdd_destroy(omp_or_bdd);
    obdd_destroy(omp_not_bdd);
#endif

#ifdef OBDD_ENABLE_CUDA
    obdd_cuda_free_device(cuda_bdd1);
    obdd_cuda_free_device(cuda_bdd2);
    obdd_cuda_free_device(cuda_and);
    obdd_cuda_free_device(cuda_or);
    obdd_cuda_free_device(cuda_xor);
    obdd_cuda_free_device(cuda_not);
#endif
}

TEST_F(BackendComparisonTest, PerformanceBenchmark) {
    const int num_vars = 10;
    int order[num_vars];
    for (int i = 0; i < num_vars; i++) order[i] = i;
    
    OBDD* bdd1 = obdd_create(num_vars, order);
    OBDD* bdd2 = obdd_create(num_vars, order);
    
    bdd1->root = build_demo_bdd(order, num_vars);
    bdd2->root = build_demo_bdd(order, num_vars);
    
    // Sequential timing
    auto seq_start = std::chrono::high_resolution_clock::now();
    OBDDNode* seq_result = obdd_apply(bdd1, bdd2, OBDD_AND);
    auto seq_end = std::chrono::high_resolution_clock::now();
    auto seq_duration = std::chrono::duration_cast<std::chrono::microseconds>(seq_end - seq_start);
    
#ifdef OBDD_ENABLE_OPENMP
    // OpenMP timing  
    auto omp_start = std::chrono::high_resolution_clock::now();
    OBDDNode* omp_result = obdd_parallel_and_omp(bdd1, bdd2);
    auto omp_end = std::chrono::high_resolution_clock::now();
    auto omp_duration = std::chrono::duration_cast<std::chrono::microseconds>(omp_end - omp_start);
    
    std::cout << "Performance comparison:" << std::endl;
    std::cout << "  Sequential: " << seq_duration.count() << " μs" << std::endl;
    std::cout << "  OpenMP: " << omp_duration.count() << " μs" << std::endl;
    
    EXPECT_NE(omp_result, nullptr);
#endif

#ifdef OBDD_ENABLE_CUDA  
    void* cuda_bdd1 = obdd_cuda_copy_to_device(bdd1);
    void* cuda_bdd2 = obdd_cuda_copy_to_device(bdd2);
    
    // CUDA timing (including transfer)
    auto cuda_start = std::chrono::high_resolution_clock::now();
    void* cuda_result = nullptr;
    obdd_cuda_and(cuda_bdd1, cuda_bdd2, &cuda_result);
    auto cuda_end = std::chrono::high_resolution_clock::now();
    auto cuda_duration = std::chrono::duration_cast<std::chrono::microseconds>(cuda_end - cuda_start);
    
    std::cout << "  CUDA: " << cuda_duration.count() << " μs" << std::endl;
    
    EXPECT_NE(cuda_result, nullptr);
    
    obdd_cuda_free_device(cuda_bdd1);
    obdd_cuda_free_device(cuda_bdd2);
    obdd_cuda_free_device(cuda_result);
#endif
    
    EXPECT_NE(seq_result, nullptr);
    
    obdd_destroy(bdd1);
    obdd_destroy(bdd2);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}