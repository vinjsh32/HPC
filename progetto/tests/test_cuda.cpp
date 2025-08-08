#include "obdd.hpp"
#include "obdd_cuda.hpp"
#include <gtest/gtest.h>
#include <vector>
#include <cstdlib>

#ifdef OBDD_ENABLE_CUDA
#include <cuda_runtime.h>
#include "obdd_cuda_types.cuh"

static int eval_device_bdd(void* dHandle, const int* assignment)
{
    DeviceOBDD dev{};
    cudaMemcpy(&dev, dHandle, sizeof(DeviceOBDD), cudaMemcpyDeviceToHost);
    std::vector<NodeGPU> nodes(dev.size);
    cudaMemcpy(nodes.data(), dev.nodes, sizeof(NodeGPU) * dev.size,
               cudaMemcpyDeviceToHost);
    int idx = 2; /* root index after flatten */
    while (nodes[idx].var >= 0) {
        int var = nodes[idx].var;
        idx = assignment[var] ? nodes[idx].high : nodes[idx].low;
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

    void *dAnd=nullptr,*dOr=nullptr,*dNot=nullptr;
    obdd_cuda_and(dA, dB, &dAnd);
    obdd_cuda_or (dA, dB, &dOr );
    obdd_cuda_not(dB, &dNot);

    void* dSelfAnd=nullptr; obdd_cuda_and(dA, dA, &dSelfAnd);
    DeviceOBDD tmp{};
    cudaMemcpy(&tmp, dSelfAnd, sizeof(DeviceOBDD), cudaMemcpyDeviceToHost);
    EXPECT_EQ(tmp.size, 3);

    void* dSelfXor=nullptr; obdd_cuda_xor(dA, dA, &dSelfXor);
    cudaMemcpy(&tmp, dSelfXor, sizeof(DeviceOBDD), cudaMemcpyDeviceToHost);
    EXPECT_EQ(tmp.size, 2);

    int assign1[3] = {1,1,0};
    EXPECT_EQ(eval_device_bdd(dAnd, assign1), 1);
    EXPECT_EQ(eval_device_bdd(dOr,  assign1), 1);
    EXPECT_EQ(eval_device_bdd(dNot, assign1), 0);

    int v[8] = {7,3,5,0,2,6,1,4};
    obdd_cuda_var_ordering(v, 8);
    for (int i = 1; i < 8; ++i)
        EXPECT_LE(v[i-1], v[i]);

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
    EXPECT_EQ(dTooLarge, nullptr);
    obdd_cuda_free_device(dBigA);
    obdd_cuda_free_device(dBigB);
    obdd_destroy(bigA);
    obdd_destroy(bigB);

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
