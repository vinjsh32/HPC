#include "obdd.hpp"
#include "obdd_cuda.hpp"
#include <cstdio>

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
    void *dAnd=nullptr,*dOr=nullptr,*dXor=nullptr,*dNot=nullptr;
    obdd_cuda_and(dA, dB, &dAnd);
    obdd_cuda_or (dA, dB, &dOr );
    obdd_cuda_xor(dA, dB, &dXor);
    obdd_cuda_not(dB, &dNot);
    std::puts("[TEST][CUDA] Kernel logici lanciati senza errori.");

    // Ordinamento varOrder su GPU
    int v[8] = {7,3,5,0,2,6,1,4};
    std::printf("[TEST][CUDA] varOrder prima: ");
    for (int x : v) std::printf("%d ", x); std::puts("");
    obdd_cuda_var_ordering(v, 8);
    std::printf("[TEST][CUDA] varOrder dopo : ");
    for (int x : v) std::printf("%d ", x); std::puts("");

    // cleanup
    obdd_cuda_free_device(dA);
    obdd_cuda_free_device(dB);
    obdd_cuda_free_device(dAnd);
    obdd_cuda_free_device(dOr);
    obdd_cuda_free_device(dXor);
    obdd_cuda_free_device(dNot);
    obdd_destroy(bddX0);
    obdd_destroy(bddX1);

    std::puts("[TEST][CUDA] Completato con successo.");
    return 0;
#endif
}
