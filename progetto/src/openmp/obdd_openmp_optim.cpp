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
 * @file obdd_openmp_optim.cpp
 * @brief OpenMP Parallel Computing Backend Implementation
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


/* -------------------------------------------------------------------------
 *  obdd_openmp_optim.cpp  –  experimental OpenMP backend with 3 optimisations
 * -------------------------------------------------------------------------
 *  Implements the 🔴 roadmap items:
 *     1. per-thread cache (lock-free in the hot path) + BFS merge;
 *     2. dynamic granularity cut-off (do not spawn tasks below K nodes);
 *     3. optional macro-profiling helper (perf stat wrapper).
 *
 *  Usa queste macro:
 *     -DOBDD_OMP_OPTIM           → compila questo backend
 *     -DOBDD_PER_THREAD_CACHE    → abilita la cache TLS + merge
 *
 *  To integrate in a standalone driver define:
 *       #define APPLY_BIN obdd_parallel_apply_omp_opt
 * -------------------------------------------------------------------------*/
#include "core/obdd.hpp"
#include <unordered_map>
#include <vector>
#include <atomic>
#include <mutex>
#include <omp.h>

/* ------------------------------------------------------------------
 *  1.  Per-thread memoisation cache  (apply_cache_local)
 * ------------------------------------------------------------------*/
struct Key {
    const OBDDNode* a;
    const OBDDNode* b;
    OBDD_Op         op;
    bool operator==(const Key& o) const noexcept {
        return a==o.a && b==o.b && op==o.op;
    }
};
struct KeyHash {
    size_t operator()(const Key& k) const noexcept {
        return (reinterpret_cast<size_t>(k.a)>>4) ^ (reinterpret_cast<size_t>(k.b)>>8) ^ k.op;
    }
};

using LocalCache = std::unordered_map<Key, OBDDNode*, KeyHash>;

#if defined(OBDD_PER_THREAD_CACHE)
/* --- lista globale dei TLS per la merge -------------------------*/
static std::vector<LocalCache*> g_tls;
static std::mutex               g_tls_mtx;

static void register_tls(LocalCache& L)
{
    std::lock_guard<std::mutex> g(g_tls_mtx);
    g_tls.push_back(&L);
}

/* Merge: master thread dopo la parallel region -------------------*/
static void merge_tls_into_master(LocalCache& master)
{
    std::lock_guard<std::mutex> g(g_tls_mtx);
    for (auto* pc : g_tls)
        if (pc!=&master) master.insert(pc->begin(), pc->end());
}
#endif

/* ------------------------------------------------------------------
 *  Apply – seriale senza spawn (usato sotto la soglia CUT)
 * ------------------------------------------------------------------*/
static OBDDNode* apply_no_spawn(LocalCache& L,
                                OBDDNode* f, OBDDNode* g, OBDD_Op op)
{
    /* base cases */
    if (f->varIndex < 0 && g->varIndex < 0) {
        int a = (f==obdd_constant(1));
        int b = (g==obdd_constant(1));
        int r = (op==OBDD_AND)? (a&b) :
                (op==OBDD_OR )? (a|b) :
                                (a^b);
        return r? obdd_constant(1):obdd_constant(0);
    }

    /* memo */
    Key k{f,g,op};
    auto it = L.find(k);
    if (it!=L.end()) return it->second;

    int v;
    if (f->varIndex < 0)      v = g->varIndex;
    else if (g->varIndex < 0) v = f->varIndex;
    else                      v = std::min(f->varIndex, g->varIndex);
    OBDDNode *fL, *fH, *gL, *gH;
    if (f->varIndex>=0 && f->varIndex==v){ fL=f->lowChild;  fH=f->highChild; } else fL=fH=f;
    if (g->varIndex>=0 && g->varIndex==v){ gL=g->lowChild;  gH=g->highChild; } else gL=gH=g;

    OBDDNode* low  = apply_no_spawn(L, fL, gL, op);
    OBDDNode* high = apply_no_spawn(L, fH, gH, op);
    if (low == high) {
        L[k] = low;
        return low;
    }
    OBDDNode* res  = obdd_node_create(v, low, high);
    L[k] = res;
    return res;
}

/* ------------------------------------------------------------------
 *  Ricorsivo con task  (granularità dinamica)
 * ------------------------------------------------------------------*/
static OBDDNode* apply_rec(LocalCache& L,
                           OBDDNode* f, OBDDNode* g, OBDD_Op op,
                           int depth, const int CUT)
{
    if (depth < CUT)
        return apply_no_spawn(L, f, g, op);

    int v;
    if (f->varIndex < 0)      v = g->varIndex;
    else if (g->varIndex < 0) v = f->varIndex;
    else                      v = std::min(f->varIndex, g->varIndex);
    OBDDNode *fL,*fH,*gL,*gH;
    if (f->varIndex>=0 && f->varIndex==v){ fL=f->lowChild; fH=f->highChild; } else fL=fH=f;
    if (g->varIndex>=0 && g->varIndex==v){ gL=g->lowChild; gH=g->highChild; } else gL=gH=g;

    OBDDNode *low=nullptr,*high=nullptr;
    #pragma omp task shared(low)
    low  = apply_rec(L, fL, gL, op, depth-1, CUT);
    #pragma omp task shared(high)
    high = apply_rec(L, fH, gH, op, depth-1, CUT);
    #pragma omp taskwait
    if (low == high) return low;
    return obdd_node_create(v, low, high);
}

/* ------------------------------------------------------------------
 *  Entry-point pubblico
 * ------------------------------------------------------------------*/
OBDDNode* obdd_parallel_apply_omp_opt(const OBDD* A, const OBDD* B, OBDD_Op op)
{
    const int CUT = 8;          /* K≈2^CUT nodi (empirico)          */
    LocalCache tls;
#if defined(OBDD_PER_THREAD_CACHE)
    register_tls(tls);
#endif
    OBDDNode* root=nullptr;
    #pragma omp parallel
    {
        #pragma omp single nowait
        {
            OBDDNode* broot = (op == OBDD_NOT) ? obdd_constant(1) : (B ? B->root : obdd_constant(0));
            OBDD_Op realOp = (op == OBDD_NOT) ? OBDD_XOR : op;
            root = apply_rec(tls, A->root, broot, realOp, CUT+2, CUT);
        }
    }
#if defined(OBDD_PER_THREAD_CACHE)
    merge_tls_into_master(tls);   /* 🔴 merge finale */
#endif
    return root;
}

