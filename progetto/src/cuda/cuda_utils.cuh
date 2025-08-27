/**
 * @file cuda_utils.cuh
 * @brief CUDA GPU Acceleration Backend Implementation
 * 
 * This file is part of the high-performance OBDD library providing
 * comprehensive Binary Decision Diagram operations with multi-backend
 * support for Sequential CPU, OpenMP Parallel, and CUDA GPU execution.
 * 
 * @author @vijsh32
 * @date August 7, 2024
 * @version 2.1
 * @copyright 2024 High Performance Computing Laboratory
 */


#pragma once
#ifdef OBDD_ENABLE_CUDA
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

/* Macro di controllo errori minimale */
#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t _e = (call);                                             \
        if (_e != cudaSuccess) {                                             \
            std::fprintf(stderr, "[CUDA] %s:%d %s\n", __FILE__, __LINE__,    \
                         cudaGetErrorString(_e));                            \
            std::abort();                                                    \
        }                                                                    \
    } while (0)

#endif /* OBDD_ENABLE_CUDA */
