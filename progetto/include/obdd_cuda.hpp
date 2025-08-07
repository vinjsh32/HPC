#pragma once
#ifndef OBDD_CUDA_HPP
#define OBDD_CUDA_HPP

#include "obdd.hpp"

/*
 * Backend CUDA pubblico (linkage C): copia Host→Device, free,
 * operatori logici binari/unari e un wrapper unico obdd_cuda_apply.
 *
 * Tutte le API sono disponibili solo se si compila con OBDD_ENABLE_CUDA.
 * Ogni operazione logica restituisce una ROBDD ridotta copiando
 * temporaneamente il risultato su host e invocando obdd_reduce().
 */

#ifdef __cplusplus
extern "C" {
#endif

#ifdef OBDD_ENABLE_CUDA

/* ---- upload & free ------------------------------------------------ */
void* obdd_cuda_copy_to_device(const OBDD* hostBDD);
/* dHandle è il valore ritornato da obdd_cuda_copy_to_device(...) */
void  obdd_cuda_free_device(void* dHandle);

/* ---- operazioni logiche ------------------------------------------ */
void obdd_cuda_and(void* dA, void* dB, void** dOut);
void obdd_cuda_or (void* dA, void* dB, void** dOut);
void obdd_cuda_xor(void* dA, void* dB, void** dOut);
void obdd_cuda_not(void* dA,           void** dOut);

/* ---- wrapper unico ------------------------------------------------ */
void* obdd_cuda_apply(void* dA, void* dB, OBDD_Op op);

/* ---- ordinamento variabili (bubble sort GPU) --------------------- */
void obdd_cuda_var_ordering(int* hostVarOrder, int n);

#endif /* OBDD_ENABLE_CUDA */

#ifdef __cplusplus
}
#endif

#endif /* OBDD_CUDA_HPP */
