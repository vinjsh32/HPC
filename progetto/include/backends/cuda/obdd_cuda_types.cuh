/**
 * @file obdd_cuda_types.cuh
 * @brief CUDA GPU Acceleration Backend Implementation
 * 
 * This file is part of the high-performance OBDD library providing
 * comprehensive Binary Decision Diagram operations with multi-backend
 * support for Sequential CPU, OpenMP Parallel, and CUDA GPU execution.
 * 
 * @author @vijsh32
 * @date August 5, 2024
 * @version 2.1
 * @copyright 2024 High Performance Computing Laboratory
 */


#pragma once
#ifdef OBDD_ENABLE_CUDA
/**
 * Tipi interni usati dal backend CUDA.
 * Non includere questo file se non si compila il backend GPU.
 */

#ifdef __CUDACC__
struct __align__(16) NodeGPU {
#else
struct alignas(16) NodeGPU {
#endif
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
