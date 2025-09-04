/*
 * This file is part of the High-Performance OBDD Library
 * Copyright (C) 2024 High Performance Computing Laboratory
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <https://www.gnu.org/licenses/>.
 * 
 * Authors: Vincenzo Ferraro
 * Student ID: 0622702113
 * Email: v.ferraro5@studenti.unisa.it
 * Assignment: Final Project - Parallel OBDD Implementation
 * Course: High Performance Computing - Prof. Moscato
 * University: Università degli studi di Salerno - Ingegneria Informatica magistrale
 */

/**
 * @file unique_table.hpp
 * @brief Unique Table Interface for OBDD Canonical Node Management
 * 
 * Corso di High Performance Computing - Prof. Moscato - Università degli studi di Salerno - Ingegneria Informatica magistrale
 * 
 * UNIQUE TABLE INTERFACE SPECIFICATION:
 * ====================================
 * This header defines the interface for the unique table data structure that
 * ensures canonical ROBDD representation. The unique table guarantees that
 * each distinct Boolean function has exactly one representation, enabling
 * structural sharing and efficient equivalence testing.
 * 
 * MULTI-PLATFORM SUPPORT:
 * =======================
 * The interface is designed for both host (CPU) and device (GPU) compilation:
 * - Host-only functions for current CPU implementations
 * - Device-compatible hash functions for future CUDA integration
 * - Conditional compilation macros for seamless multi-target builds
 * 
 * CANONICAL REPRESENTATION GUARANTEE:
 * ===================================
 * - Each unique (variable, low_child, high_child) triple maps to single node
 * - Structural sharing maximizes memory efficiency through deduplication
 * - Automatic elimination of isomorphic subtrees
 * - O(1) equivalence testing through pointer comparison
 * 
 * @author vinjsh32
 * @date September 2, 2024
 * @version 3.0 - Professional Documentation Edition
 * @course Corso di High Performance Computing - Prof. Moscato
 * @university Università degli studi di Salerno - Ingegneria Informatica magistrale
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
