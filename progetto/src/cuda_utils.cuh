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
