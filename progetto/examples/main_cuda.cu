/**
 *  main_cuda.cu
 *  ------------
 *  Collaudo rapido del backend CUDA:
 *    – costruzione di due BDD semplici (x0, x1)
 *    – copy_to_device
 *    – kernel AND / OR / XOR / NOT
 *    – kernel bubble-sort su un vettore varOrder (host → device → host)
 *
 *  Compilazione (esempio NVCC):
 *      nvcc -std=c++17 -O2 \
 *          obdd_core.cpp  obdd_cuda.cu  main_cuda.cu  -o test_cuda
 */

#include "obdd.hpp"
#include "obdd_cuda.hpp"
#include <cstdio>
#include <cuda_runtime.h>

/* helper cuda-error-check minimale */
#define CUDA_CHECK(x)  do {                                 \
    cudaError_t e = (x);                                    \
    if (e != cudaSuccess){                                  \
        printf("[CUDA] %s:%d %s\n", __FILE__, __LINE__,     \
               cudaGetErrorString(e)); return 1; } } while(0)

int main()
{
    /* --------- 1) costruiamo due BDD unari: x0, x1 ------------- */
    int order[3] = {0,1,2};
    OBDD* bddX0 = obdd_create(3, order);
    OBDD* bddX1 = obdd_create(3, order);
    bddX0->root = obdd_node_create(0, obdd_constant(0), obdd_constant(1));
    bddX1->root = obdd_node_create(1, obdd_constant(0), obdd_constant(1));

    /* --------- 2) copia su device ------------------------------ */
    void* dA = obdd_cuda_copy_to_device(bddX0);
    void* dB = obdd_cuda_copy_to_device(bddX1);

    /* --------- 3) test kernel AND / OR / XOR ------------------- */
    void *dAnd=nullptr,*dOr=nullptr,*dXor=nullptr,*dNot=nullptr;
    obdd_cuda_and(dA, dB, &dAnd);   CUDA_CHECK(cudaDeviceSynchronize());
    obdd_cuda_or (dA, dB, &dOr );   CUDA_CHECK(cudaDeviceSynchronize());
    obdd_cuda_xor(dA, dB, &dXor);   CUDA_CHECK(cudaDeviceSynchronize());
    obdd_cuda_not(dB, &dNot);       CUDA_CHECK(cudaDeviceSynchronize());
    printf("[CUDA] Kernel logici lanciati senza errori.\n");

    /* --------- 4) bubble-sort GPU su un array ------------------ */
    int v[8] = {7,3,5,0,2,6,1,4};
    printf("[CUDA] varOrder prima: ");
    for(int x: v) printf("%d ", x); puts("");
    obdd_cuda_var_ordering(v, 8);
    printf("[CUDA] varOrder dopo : ");
    for(int x: v) printf("%d ", x); puts("");

    /* --------- 5) cleanup -------------------------------------- */
    obdd_cuda_free_device(dA);
    obdd_cuda_free_device(dB);
    obdd_cuda_free_device(dAnd);
    obdd_cuda_free_device(dOr);
    obdd_cuda_free_device(dXor);
    obdd_cuda_free_device(dNot);
    obdd_destroy(bddX0);
    obdd_destroy(bddX1);

    printf("[CUDA] Test completato con successo.\n");
    return 0;
}
