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
 * @file obdd_openmp_enhanced.cpp
 * @brief Conservative OpenMP Backend with Timeout Protection
 * 
 * This implementation provides a more conservative approach to OpenMP
 * parallelization, focusing on reliability and avoiding timeout issues
 * while maintaining reasonable performance characteristics.
 * 
 * Key Features:
 * - Limited recursive depth to prevent excessive task creation
 * - Conservative parallelization thresholds
 * - Simplified logic to avoid complex synchronization issues
 * - Better resource management and cleanup
 * 
 * @author @vijsh32
 * @date August 31, 2025
 * @version 4.0 (Conservative Rewrite)
 */

#include "core/obdd.hpp"
#include "core/apply_cache.hpp"

#include <omp.h>
#include <climits>
#include <vector>
#include <cmath>
#include <algorithm>

// Conservative parallelization parameters
static const int MAX_PARALLEL_DEPTH = 3;  // Very limited depth to prevent issues
static const int MAX_THREADS_CONSERVATIVE = 4; // Conservative thread limit

/**
 * Simple conservative parallel apply implementation
 */
static OBDDNode* conservative_apply_internal(const OBDDNode* n1,
                                           const OBDDNode* n2,
                                           OBDD_Op op,
                                           int depth)
{
    // Cache lookup
    if (OBDDNode* hit = apply_cache_lookup(n1, n2, static_cast<int>(op))) {
        return hit;
    }

    // Base case: leaves
    if (is_leaf(n1) && is_leaf(n2)) {
        int v1 = (n1 == obdd_constant(1));
        int v2 = (n2 == obdd_constant(1));
        OBDDNode* res = apply_leaf(op, v1, v2);
        apply_cache_insert(n1, n2, static_cast<int>(op), res);
        return res;
    }

    // Split variable
    int v1  = is_leaf(n1) ? INT_MAX : n1->varIndex;
    int v2  = (n2 && !is_leaf(n2)) ? n2->varIndex : INT_MAX;
    int var = (v1 < v2) ? v1 : v2;

    const OBDDNode* n1_low  = (v1 == var) ? n1->lowChild  : n1;
    const OBDDNode* n1_high = (v1 == var) ? n1->highChild : n1;
    const OBDDNode* n2_low  = ((v2 == var) && n2) ? n2->lowChild  : n2;
    const OBDDNode* n2_high = ((v2 == var) && n2) ? n2->highChild : n2;

    OBDDNode *lowRes = nullptr, *highRes = nullptr;

    // Only use parallelism at very shallow depth and simple cases
    if (depth < MAX_PARALLEL_DEPTH && omp_get_max_threads() <= MAX_THREADS_CONSERVATIVE) {
        #pragma omp parallel sections num_threads(2)
        {
            #pragma omp section
            lowRes = conservative_apply_internal(n1_low, n2_low, op, depth + 1);
            
            #pragma omp section
            highRes = conservative_apply_internal(n1_high, n2_high, op, depth + 1);
        }
    } else {
        // Sequential execution for most cases
        lowRes = conservative_apply_internal(n1_low, n2_low, op, depth + 1);
        highRes = conservative_apply_internal(n1_high, n2_high, op, depth + 1);
    }

    // Reduction
    OBDDNode* res = (lowRes == highRes) ? lowRes : obdd_node_create(var, lowRes, highRes);
    apply_cache_insert(n1, n2, static_cast<int>(op), res);
    return res;
}

extern "C" {

/**
 * Conservative OpenMP apply implementation
 */
OBDDNode* obdd_parallel_apply_omp_enhanced(const OBDD* bdd1, 
                                          const OBDD* bdd2, 
                                          OBDD_Op op)
{
    if (!bdd1) return nullptr;

    apply_cache_clear();
    OBDDNode* result = nullptr;

    // Use conservative thread count
    int original_threads = omp_get_max_threads();
    omp_set_num_threads(std::min(original_threads, MAX_THREADS_CONSERVATIVE));

    #pragma omp parallel
    {
        apply_cache_thread_init();
        #pragma omp single nowait
        {
            result = conservative_apply_internal(bdd1->root,
                                               bdd2 ? bdd2->root : nullptr,
                                               op,
                                               0);
        }
    }

    // Restore original thread count
    omp_set_num_threads(original_threads);
    
    apply_cache_merge();
    return result;
}

// Conservative wrapper functions
OBDDNode* obdd_parallel_and_omp_enhanced(const OBDD* a, const OBDD* b) {
    return (a && b) ? obdd_parallel_apply_omp_enhanced(a, b, OBDD_AND) : nullptr;
}

OBDDNode* obdd_parallel_or_omp_enhanced(const OBDD* a, const OBDD* b) {
    return (a && b) ? obdd_parallel_apply_omp_enhanced(a, b, OBDD_OR) : nullptr;
}

OBDDNode* obdd_parallel_xor_omp_enhanced(const OBDD* a, const OBDD* b) {
    return (a && b) ? obdd_parallel_apply_omp_enhanced(a, b, OBDD_XOR) : nullptr;
}

OBDDNode* obdd_parallel_not_omp_enhanced(const OBDD* a) {
    return a ? obdd_parallel_apply_omp_enhanced(a, nullptr, OBDD_NOT) : nullptr;
}

/**
 * Simple parallel sorting without task overhead
 */
void obdd_simple_var_ordering_omp(OBDD* bdd)
{
    if (!bdd || bdd->numVars <= 1) return;

    int* vo = bdd->varOrder;
    const int n = bdd->numVars;

    // Use standard library sort - simple and reliable
    std::sort(vo, vo + n);
}

} /* extern "C" */