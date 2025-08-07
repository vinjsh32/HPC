#pragma once
#ifdef OBDD_ENABLE_CUDA
/**
 * Tipi interni usati dal backend CUDA.
 * Non includere questo file se non si compila il backend GPU.
 */

struct __align__(16) NodeGPU {
    int var;   // -1 = leaf
    int low;   // index of low child (or 0/1 for constants)
    int high;  // index of high child (or 0/1 for constants)
};

struct DeviceOBDD {
    NodeGPU* nodes;   // device pointer
    int      size;    // #nodes
    int      nVars;   // #boolean vars
};

/* Thread per block "default" (puoi modificarlo in cmake via -DTPB=...) */
#ifndef OBDD_CUDA_TPB
#define OBDD_CUDA_TPB 128
#endif

#endif /* OBDD_ENABLE_CUDA */
