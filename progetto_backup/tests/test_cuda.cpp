#include "obdd.hpp"
#include "obdd_cuda.hpp"
#include <gtest/gtest.h>
#include <vector>
#include <cstdlib>

#ifdef OBDD_ENABLE_CUDA
#include <cuda_runtime.h>
#include "obdd_cuda_types.cuh"

static void debug_device_bdd(void* dHandle, const char* name) 
{
    DeviceOBDD dev{};
    cudaMemcpy(&dev, dHandle, sizeof(DeviceOBDD), cudaMemcpyDeviceToHost);
    std::vector<NodeGPU> nodes(dev.size);
    cudaMemcpy(nodes.data(), dev.nodes, sizeof(NodeGPU) * dev.size,
               cudaMemcpyDeviceToHost);
    
    printf("=== BDD %s (size=%d, nVars=%d) ===\n", name, dev.size, dev.nVars);
    for (int i = 0; i < dev.size; ++i) {
        printf("Node[%d]: var=%d, low=%d, high=%d\n", 
               i, nodes[i].var, nodes[i].low, nodes[i].high);
    }
    printf("\n");
}

static int eval_device_bdd(void* dHandle, const int* assignment)
{
    DeviceOBDD dev{};
    cudaMemcpy(&dev, dHandle, sizeof(DeviceOBDD), cudaMemcpyDeviceToHost);
    std::vector<NodeGPU> nodes(dev.size);
    cudaMemcpy(nodes.data(), dev.nodes, sizeof(NodeGPU) * dev.size,
               cudaMemcpyDeviceToHost);
    
    if (dev.size <= 2) {
        if (dev.size == 2) {
            return nodes[0].low;
        }
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
            if (nodes[i].var == -1) {
                return nodes[i].low;
            }
        }
        if (dev.size > 1) {
            return nodes[dev.size - 1].low;
        }
        return 0;
    }
    
    int idx = rootIdx;
    while (idx >= 0 && idx < dev.size && nodes[idx].var >= 0) {
        int var = nodes[idx].var;
        if (var >= dev.nVars) {
            return 0;
        }
        
        int next_idx = assignment[var] ? nodes[idx].high : nodes[idx].low;
        
        if (next_idx < 0 || next_idx >= dev.size) {
            return 0;
        }
        
        idx = next_idx;
    }
    
    if (idx < 0 || idx >= dev.size) {
        return 0;
    }
    
    return nodes[idx].low;
}

TEST(CUDABackend, BasicOperations)
{
    int order[3] = {0,1,2};
    OBDD* bddX0 = obdd_create(3, order);
    OBDD* bddX1 = obdd_create(3, order);
    bddX0->root = obdd_node_create(0, obdd_constant(0), obdd_constant(1));
    bddX1->root = obdd_node_create(1, obdd_constant(0), obdd_constant(1));

    void* dA = obdd_cuda_copy_to_device(bddX0);
    void* dB = obdd_cuda_copy_to_device(bddX1);

    debug_device_bdd(dA, "dA (x0)");
    debug_device_bdd(dB, "dB (x1)");

    void *dAnd=nullptr,*dOr=nullptr,*dNot=nullptr;
    obdd_cuda_and(dA, dB, &dAnd);
    obdd_cuda_or (dA, dB, &dOr );
    obdd_cuda_not(dB, &dNot);

    debug_device_bdd(dAnd, "dAnd");
    debug_device_bdd(dOr, "dOr");
    debug_device_bdd(dNot, "dNot");

    void* dSelfAnd=nullptr; obdd_cuda_and(dA, dA, &dSelfAnd);
    void* dSelfXor=nullptr; obdd_cuda_xor(dA, dA, &dSelfXor);
    
    debug_device_bdd(dSelfAnd, "dSelfAnd");
    debug_device_bdd(dSelfXor, "dSelfXor");
    

    int assign1[3] = {1,1,0};
    int andResult = eval_device_bdd(dAnd, assign1);
    int orResult = eval_device_bdd(dOr, assign1);
    int notResult = eval_device_bdd(dNot, assign1);
    
    printf("Results for assign1 {1,1,0}: AND=%d, OR=%d, NOT=%d\n", andResult, orResult, notResult);
    
    int assign2[3] = {0,0,0};
    int andResult2 = eval_device_bdd(dAnd, assign2);
    int orResult2 = eval_device_bdd(dOr, assign2);
    int notResult2 = eval_device_bdd(dNot, assign2);
    
    printf("Results for assign2 {0,0,0}: AND=%d, OR=%d, NOT=%d\n", andResult2, orResult2, notResult2);
    
    int selfAndResult = eval_device_bdd(dSelfAnd, assign1);
    int selfXorResult = eval_device_bdd(dSelfXor, assign1);
    
    printf("Self-operations for assign1: SelfAND=%d, SelfXOR=%d\n", selfAndResult, selfXorResult);
    
    EXPECT_EQ(andResult, 1) << "AND(x0=1, x1=1) should be 1";
    EXPECT_EQ(orResult, 1) << "OR(x0=1, x1=1) should be 1";
    EXPECT_EQ(notResult, 0) << "NOT(x1=1) should be 0";
    EXPECT_EQ(andResult2, 0) << "AND(x0=0, x1=0) should be 0";
    EXPECT_EQ(orResult2, 0) << "OR(x0=0, x1=0) should be 0";
    EXPECT_EQ(notResult2, 1) << "NOT(x1=0) should be 1";
    EXPECT_EQ(selfAndResult, 1) << "x0 AND x0 with x0=1 should be 1";
    EXPECT_EQ(selfXorResult, 0) << "x0 XOR x0 should always be 0";

    int v[8] = {7,3,5,0,2,6,1,4};
    obdd_cuda_var_ordering(v, 8);
    for (int i = 1; i < 8; ++i)
        EXPECT_LE(v[i-1], v[i]);

    obdd_cuda_free_device(dA);
    obdd_cuda_free_device(dB);
    obdd_cuda_free_device(dAnd);
    obdd_cuda_free_device(dOr);
    obdd_cuda_free_device(dNot);
    obdd_cuda_free_device(dSelfAnd);
    obdd_cuda_free_device(dSelfXor);
    obdd_destroy(bddX0);
    obdd_destroy(bddX1);
}

TEST(CUDABackend, MemoryLimitTest)
{
    int order[3] = {0,1,2};
    setenv("OBDD_CUDA_MAX_PAIRS", "16", 1);
    
    OBDD* bigA = obdd_create(3, order);
    OBDD* bigB = obdd_create(3, order);
    
    OBDDNode* a2 = obdd_node_create(2, obdd_constant(0), obdd_constant(1));
    OBDDNode* a1 = obdd_node_create(1, obdd_constant(0), a2);
    bigA->root = obdd_node_create(0, a1, a2);
    
    OBDDNode* b2 = obdd_node_create(2, obdd_constant(0), obdd_constant(1));
    OBDDNode* b1 = obdd_node_create(1, obdd_constant(0), b2);
    bigB->root = obdd_node_create(0, b1, b2);
    
    void* dBigA = obdd_cuda_copy_to_device(bigA);
    void* dBigB = obdd_cuda_copy_to_device(bigB);
    
    void* dTooLarge = nullptr;
    obdd_cuda_and(dBigA, dBigB, &dTooLarge);
    
    EXPECT_EQ(dTooLarge, nullptr) << "Operation should fail when exceeding memory limit";
    
    obdd_cuda_free_device(dBigA);
    obdd_cuda_free_device(dBigB);
    obdd_destroy(bigA);
    obdd_destroy(bigB);
}

#else
TEST(CUDABackend, DisabledBackend) {
    GTEST_SKIP() << "Backend CUDA disabilitato: compila con CUDA=1.";
}
#endif

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
