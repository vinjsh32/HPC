/**
 * @file unique_table.hpp
 * @brief OBDD Library Implementation Component
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
#ifndef UNIQUE_TABLE_HPP
#define UNIQUE_TABLE_HPP

#include "obdd.hpp"
#include <stdint.h>

/* --------------------------------------------------------------------------
 *  Unique table support (host + device)
 *  La tabella garantisce canonicità dei nodi (var,low,high) → nodo unico.
 *  Può essere usata sia nel core sequenziale che in future versioni CUDA;
 *  per questo le funzioni hash sono marcate __host__ __device__.
 * --------------------------------------------------------------------------*/

#ifdef __CUDACC__
#define OBDD_HD __host__ __device__
#else
#define OBDD_HD
#endif

/* dimensione di default; se serve si può rendere dinamica */
#ifndef UNIQUE_SIZE
#define UNIQUE_SIZE 10007
#endif

typedef struct {
    int         var;   /* indice variabile */
    OBDDNode*   low;   /* figlio 0  */
    OBDDNode*   high;  /* figlio 1  */
    OBDDNode*   result;/* puntatore unico al nodo canonico */
} UniqueEntry;

/* tabella globale (host) – la definizione vera è in unique_table.cpp */
extern UniqueEntry unique_table[UNIQUE_SIZE];

/* --------------------------------------------------------------------------
 *  Funzioni hash / utilità
 * --------------------------------------------------------------------------*/

/**
 * Hash su tripla (var, low, high) – uguale in host e device.
 */
OBDD_HD static inline size_t triple_hash(int var,
                                         const OBDDNode* low,
                                         const OBDDNode* high)
{
    uintptr_t l = (uintptr_t)low  >> 3;
    uintptr_t h = (uintptr_t)high >> 3;
    return (l ^ h ^ (uintptr_t)var) % UNIQUE_SIZE;
}

#ifdef __cplusplus
extern "C" {
#endif

/* resetta tutta la unique – solo host */
void unique_table_clear(void);

/* lookup/insert (host‑only per ora – versione device in futuro) */
OBDDNode* unique_table_get_or_create(int var,
                                     OBDDNode* low,
                                     OBDDNode* high);

#ifdef __cplusplus
}
#endif

#undef OBDD_HD

#endif /* UNIQUE_TABLE_HPP */
