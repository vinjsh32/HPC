#include "obdd.hpp"
#include "obdd_cuda.hpp"
#include <cstdio>
#include <vector>
#include <cassert>

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
#endif

int main()
{
#ifndef OBDD_ENABLE_CUDA
    std::printf("[TEST][CUDA] Backend CUDA disabilitato: abilita OBDD_ENABLE_CUDA (make CUDA=1).\n");
    return 0;
#else
    int order[3] = {0,1,2};

    // Costruiamo due BDD semplici: x0 e x1
    OBDD* bddX0 = obdd_create(3, order);
    OBDD* bddX1 = obdd_create(3, order);
    bddX0->root = obdd_node_create(0, obdd_constant(0), obdd_constant(1));
    bddX1->root = obdd_node_create(1, obdd_constant(0), obdd_constant(1));

    // Copia su device
    void* dA = obdd_cuda_copy_to_device(bddX0);
    void* dB = obdd_cuda_copy_to_device(bddX1);

    // Operazioni logiche
    void *dAnd=nullptr,*dOr=nullptr,*dNot=nullptr;
    obdd_cuda_and(dA, dB, &dAnd);
    obdd_cuda_or (dA, dB, &dOr );
    obdd_cuda_not(dB, &dNot);

    // Verifica riduzione: x0 AND x0 => BDD di 3 nodi
    void* dSelfAnd=nullptr; obdd_cuda_and(dA, dA, &dSelfAnd);
    DeviceOBDD tmp{};
    cudaMemcpy(&tmp, dSelfAnd, sizeof(DeviceOBDD), cudaMemcpyDeviceToHost);
    assert(tmp.size == 3);

    // x0 XOR x0 => costante 0, vettore di 2 nodi
    void* dSelfXor=nullptr; obdd_cuda_xor(dA, dA, &dSelfXor);
    cudaMemcpy(&tmp, dSelfXor, sizeof(DeviceOBDD), cudaMemcpyDeviceToHost);
    assert(tmp.size == 2);

    int assign1[3] = {1,1,0};
    assert(eval_device_bdd(dAnd, assign1) == 1);
    assert(eval_device_bdd(dOr,  assign1) == 1);
    assert(eval_device_bdd(dNot, assign1) == 0);

    // Ordinamento varOrder su GPU
    int v[8] = {7,3,5,0,2,6,1,4};
    obdd_cuda_var_ordering(v, 8);
    for (int i = 1; i < 8; ++i)
        assert(v[i-1] <= v[i]);

    // cleanup
    obdd_cuda_free_device(dA);
    obdd_cuda_free_device(dB);
      obdd_cuda_free_device(dAnd);
      obdd_cuda_free_device(dOr);
      obdd_cuda_free_device(dNot);
      obdd_cuda_free_device(dSelfAnd);
      obdd_cuda_free_device(dSelfXor);
    obdd_destroy(bddX0);
    obdd_destroy(bddX1);

    std::puts("[TEST][CUDA] Completato con successo.");
    return 0;
#endif
}
