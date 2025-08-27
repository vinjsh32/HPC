/**
 * @file test_cuda_extended.cpp 
 * @brief Extended CUDA backend tests for comprehensive coverage
 */

#include "core/obdd.hpp"
#include "cuda/obdd_cuda.hpp"
#include "advanced/obdd_advanced_math.hpp"
#include <gtest/gtest.h>
#include <vector>
#include <chrono>

#ifdef OBDD_ENABLE_CUDA
#include <cuda_runtime.h>
#include "obdd_cuda_types.cuh"

class CUDAExtendedTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Check CUDA availability
        int deviceCount;
        cudaGetDeviceCount(&deviceCount);
        if (deviceCount == 0) {
            GTEST_SKIP() << "No CUDA devices available";
        }
    }
    
    void debug_device_bdd(void* dHandle, const char* name) {
        DeviceOBDD dev{};
        cudaMemcpy(&dev, dHandle, sizeof(DeviceOBDD), cudaMemcpyDeviceToHost);
        std::vector<NodeGPU> nodes(dev.size);
        cudaMemcpy(nodes.data(), dev.nodes, sizeof(NodeGPU) * dev.size,
                   cudaMemcpyDeviceToHost);
        
        printf("=== BDD %s (size=%d, nVars=%d) ===\n", name, dev.size, dev.nVars);
        for (int i = 0; i < std::min(dev.size, 10); ++i) {
            printf("Node[%d]: var=%d, low=%d, high=%d\n", 
                   i, nodes[i].var, nodes[i].low, nodes[i].high);
        }
        if (dev.size > 10) printf("... [%d more nodes]\n", dev.size - 10);
        printf("\n");
    }
};

TEST_F(CUDAExtendedTest, ErrorHandling) {
    // Test NULL inputs safely
    void* null_result = nullptr;
    
    // These should handle NULL gracefully  
    // Note: Actual behavior may vary based on implementation
    std::cout << "Testing NULL input handling..." << std::endl;
    
    // Test obdd_cuda_free_device with NULL (should be safe)
    obdd_cuda_free_device(nullptr); // Should not crash
    
    std::cout << "NULL handling test completed" << std::endl;
}

TEST_F(CUDAExtendedTest, LargeBDDOperations) {
    const int num_vars = 8;
    int order[num_vars];
    for (int i = 0; i < num_vars; i++) order[i] = i;
    
    // Create larger BDD structure
    OBDD* bdd_large = obdd_create(num_vars, order);
    
    // Build complex BDD: chain of variables
    OBDDNode* current = obdd_constant(1);
    for (int i = num_vars - 1; i >= 0; i--) {
        current = obdd_node_create(i, obdd_constant(0), current);
    }
    bdd_large->root = current;
    
    // Test CUDA operations
    void* dLarge = obdd_cuda_copy_to_device(bdd_large);
    ASSERT_NE(dLarge, nullptr);
    
    debug_device_bdd(dLarge, "Large BDD");
    
    // Self operations
    void* dSelfAnd = nullptr, *dSelfOr = nullptr;
    obdd_cuda_and(dLarge, dLarge, &dSelfAnd);
    obdd_cuda_or(dLarge, dLarge, &dSelfOr);
    
    EXPECT_NE(dSelfAnd, nullptr);
    EXPECT_NE(dSelfOr, nullptr);
    
    obdd_cuda_free_device(dLarge);
    obdd_cuda_free_device(dSelfAnd);  
    obdd_cuda_free_device(dSelfOr);
    obdd_destroy(bdd_large);
}

TEST_F(CUDAExtendedTest, MultipleVariableOrdering) {
    int order1[5] = {0,1,2,3,4};
    int order2[5] = {4,3,2,1,0}; // Reverse order
    
    OBDD* bdd1 = obdd_create(5, order1);
    OBDD* bdd2 = obdd_create(5, order2);
    
    // Same logical function, different variable ordering
    bdd1->root = obdd_node_create(0,
        obdd_node_create(1, obdd_constant(0), obdd_constant(1)),
        obdd_node_create(2, obdd_constant(1), obdd_constant(0))
    );
    
    bdd2->root = obdd_node_create(4,
        obdd_node_create(3, obdd_constant(0), obdd_constant(1)), 
        obdd_node_create(2, obdd_constant(1), obdd_constant(0))
    );
    
    void* dBdd1 = obdd_cuda_copy_to_device(bdd1);
    void* dBdd2 = obdd_cuda_copy_to_device(bdd2);
    
    debug_device_bdd(dBdd1, "BDD1 (order 0,1,2,3,4)");
    debug_device_bdd(dBdd2, "BDD2 (order 4,3,2,1,0)");
    
    // Test operations between different orderings
    void* dResult = nullptr;
    obdd_cuda_and(dBdd1, dBdd2, &dResult);
    EXPECT_NE(dResult, nullptr);
    
    debug_device_bdd(dResult, "Cross-ordering AND result");
    
    obdd_cuda_free_device(dBdd1);
    obdd_cuda_free_device(dBdd2);
    obdd_cuda_free_device(dResult);
    obdd_destroy(bdd1);
    obdd_destroy(bdd2);
}

TEST_F(CUDAExtendedTest, PerformanceStress) {
    const int iterations = 50;
    const int num_vars = 6;
    int order[num_vars];
    for (int i = 0; i < num_vars; i++) order[i] = i;
    
    OBDD* test_bdd = obdd_create(num_vars, order);
    test_bdd->root = build_demo_bdd(order, num_vars);
    
    void* dBdd = obdd_cuda_copy_to_device(test_bdd);
    ASSERT_NE(dBdd, nullptr);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Perform many operations in succession
    std::vector<void*> results;
    for (int i = 0; i < iterations; i++) {
        void* result = nullptr;
        obdd_cuda_and(dBdd, dBdd, &result);
        if (result) results.push_back(result);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    EXPECT_LT(duration.count(), 5000); // Should complete within 5 seconds
    EXPECT_EQ(results.size(), iterations);
    
    // Cleanup
    for (void* result : results) {
        obdd_cuda_free_device(result);
    }
    obdd_cuda_free_device(dBdd);
    obdd_destroy(test_bdd);
}

TEST_F(CUDAExtendedTest, AdvancedMathCUDA) {
    // Test advanced mathematical BDDs on CUDA
    OBDD* modular_bdd = obdd_modular_pythagorean(3, 7);
    ASSERT_NE(modular_bdd, nullptr);
    
    void* dModular = obdd_cuda_copy_to_device(modular_bdd);
    ASSERT_NE(dModular, nullptr);
    
    debug_device_bdd(dModular, "Modular Pythagorean CUDA");
    
    // Test NOT operation on mathematical BDD
    void* dNotModular = nullptr;
    obdd_cuda_not(dModular, &dNotModular);
    EXPECT_NE(dNotModular, nullptr);
    
    debug_device_bdd(dNotModular, "NOT(Modular Pythagorean)");
    
    obdd_cuda_free_device(dModular);
    obdd_cuda_free_device(dNotModular);
    obdd_destroy(modular_bdd);
}

TEST_F(CUDAExtendedTest, ThrustIntegration) {
    // Test variable ordering with Thrust
    const int num_elements = 1000;
    std::vector<int> test_array(num_elements);
    
    // Fill with random values
    for (int i = 0; i < num_elements; i++) {
        test_array[i] = num_elements - i; // Reverse order
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    obdd_cuda_var_ordering(test_array.data(), num_elements);
    auto end = std::chrono::high_resolution_clock::now();
    
    // Verify sorting
    for (int i = 1; i < num_elements; i++) {
        EXPECT_LE(test_array[i-1], test_array[i]);
    }
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    EXPECT_LT(duration.count(), 10000); // Should be fast with GPU parallelism
}

TEST_F(CUDAExtendedTest, MemoryStressTest) {
    // Test behavior under memory pressure
    const int num_bdds = 20;
    int order[4] = {0,1,2,3};
    
    std::vector<void*> device_handles;
    
    for (int i = 0; i < num_bdds; i++) {
        OBDD* bdd = obdd_create(4, order);
        bdd->root = obdd_node_create(i % 4,
            obdd_node_create((i+1) % 4, obdd_constant(0), obdd_constant(1)),
            obdd_node_create((i+2) % 4, obdd_constant(1), obdd_constant(0))
        );
        
        void* dBdd = obdd_cuda_copy_to_device(bdd);
        if (dBdd) {
            device_handles.push_back(dBdd);
        }
        
        obdd_destroy(bdd);
    }
    
    EXPECT_GT(device_handles.size(), 0);
    
    // Test operations between multiple BDDs
    if (device_handles.size() >= 2) {
        void* result = nullptr;
        obdd_cuda_and(device_handles[0], device_handles[1], &result);
        if (result) {
            device_handles.push_back(result);
        }
    }
    
    // Cleanup all handles
    for (void* handle : device_handles) {
        obdd_cuda_free_device(handle);
    }
}

#else
TEST(CUDAExtendedTest, DisabledBackend) {
    GTEST_SKIP() << "CUDA backend disabled: compile with CUDA=1.";
}
#endif

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}