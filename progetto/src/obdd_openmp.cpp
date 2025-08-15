/**
 * @file obdd_openmp.cpp
 * @brief Backend parallelo OpenMP per OBDD/ROBDD.
 *
 *  – Versioni multi-thread di apply / AND-OR-XOR-NOT e var-ordering.
 *  – Condivide la apply-cache con il core, ora per-thread e senza lock.
 *  – API con linkage C (vedi obdd.hpp).
 */

#include "obdd.hpp"
#include "apply_cache.hpp"     /* memoization helpers */

#include <omp.h>
#include <climits>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cassert>

static inline int compute_task_cutoff() {
    return static_cast<int>(std::log2(omp_get_max_threads()));
}

/* --------------------------------------------------------------------------
 *  Ricorsione parallela con task
 * -------------------------------------------------------------------------- */
static OBDDNode* obdd_parallel_apply_internal(const OBDDNode* n1,
                                              const OBDDNode* n2,
                                              OBDD_Op         op,
                                              int             depth,
                                              int             task_cutoff)
{
    /* 1) memo-lookup nella cache locale */
    if (OBDDNode* hit = apply_cache_lookup(n1, n2, static_cast<int>(op)))
        return hit;

    /* 2) base case: due foglie */
    if (is_leaf(n1) && is_leaf(n2)) {
        int v1 = (n1 == obdd_constant(1));
        int v2 = (n2 == obdd_constant(1));
        OBDDNode* res = apply_leaf(op, v1, v2);
        apply_cache_insert(n1, n2, static_cast<int>(op), res);
        return res;
    }

    /* 3) variabile di split */
    int v1  = is_leaf(n1) ? INT_MAX : n1->varIndex;
    int v2  = (n2 && !is_leaf(n2)) ? n2->varIndex : INT_MAX;
    int var = (v1 < v2) ? v1 : v2;

    const OBDDNode* n1_low  = (v1 == var) ? n1->lowChild  : n1;
    const OBDDNode* n1_high = (v1 == var) ? n1->highChild : n1;
    const OBDDNode* n2_low  = ((v2 == var) && n2) ? n2->lowChild  : n2;
    const OBDDNode* n2_high = ((v2 == var) && n2) ? n2->highChild : n2;

    /* 4) task sui due rami */
    OBDDNode *lowRes  = nullptr,
             *highRes = nullptr;

    #pragma omp taskgroup
    {
        #pragma omp task shared(lowRes)  firstprivate(n1_low, n2_low, op, depth, task_cutoff) depend(out:lowRes) final(depth >= task_cutoff) if(depth < task_cutoff)
        lowRes  = obdd_parallel_apply_internal(n1_low,  n2_low,  op, depth + 1, task_cutoff);

        #pragma omp task shared(highRes) firstprivate(n1_high, n2_high, op, depth, task_cutoff) depend(out:highRes) final(depth >= task_cutoff) if(depth < task_cutoff)
        highRes = obdd_parallel_apply_internal(n1_high, n2_high, op, depth + 1, task_cutoff);
    }

    /* 5) riduzione locale */
    OBDDNode* res = (lowRes == highRes)
                    ? lowRes
                    : obdd_node_create(var, lowRes, highRes);

    /* 6) salva nella cache locale */
    apply_cache_insert(n1, n2, static_cast<int>(op), res);

    return res;
}

/* =========================================================================
 *  API OpenMP (linkage C)
 * ========================================================================= */
extern "C" {

OBDDNode* obdd_parallel_apply_omp(const OBDD* bdd1,
                                  const OBDD* bdd2,
                                  OBDD_Op     op)
{
    if (!bdd1) return nullptr;

    apply_cache_clear();
    OBDDNode* out = nullptr;
    const int task_cutoff = compute_task_cutoff();

    #pragma omp parallel
    {
        apply_cache_thread_init();
        #pragma omp single nowait
        out = obdd_parallel_apply_internal(bdd1->root,
                                           bdd2 ? bdd2->root : nullptr,
                                           op,
                                           0,
                                           task_cutoff);
    }
    apply_cache_merge();
    return out;
}

OBDDNode* obdd_parallel_and_omp(const OBDD* a, const OBDD* b)
{ return (a && b) ? obdd_parallel_apply_omp(a, b, OBDD_AND) : nullptr; }
OBDDNode* obdd_parallel_or_omp (const OBDD* a, const OBDD* b)
{ return (a && b) ? obdd_parallel_apply_omp(a, b, OBDD_OR)  : nullptr; }
OBDDNode* obdd_parallel_xor_omp(const OBDD* a, const OBDD* b)
{ return (a && b) ? obdd_parallel_apply_omp(a, b, OBDD_XOR) : nullptr; }
OBDDNode* obdd_parallel_not_omp(const OBDD* a)
{ return a ? obdd_parallel_apply_omp(a, nullptr, OBDD_NOT)  : nullptr; }

/* ------------------------------------------------------------------
 *  Merge sort parallelo sul vettore varOrder
 * ------------------------------------------------------------------ */
static void merge(int* arr, int* tmp, int l, int m, int r)
{
    int i = l, j = m, k = l;
    while (i < m && j < r)
        tmp[k++] = (arr[i] <= arr[j]) ? arr[i++] : arr[j++];
    while (i < m) tmp[k++] = arr[i++];
    while (j < r) tmp[k++] = arr[j++];
    for (int t = l; t < r; ++t) arr[t] = tmp[t];
}

static void merge_sort_serial(int* arr, int* tmp, int l, int r)
{
    if (r - l <= 1) return;
    int m = l + (r - l) / 2;
    merge_sort_serial(arr, tmp, l, m);
    merge_sort_serial(arr, tmp, m, r);
    merge(arr, tmp, l, m, r);
}

static void merge_sort_parallel(int* arr, int* tmp, int l, int r, int task_cutoff)
{
    const int n = r - l;
    if (n <= 1) return;

    const int grainsize = std::max(1, n >> task_cutoff);

    #pragma omp taskgroup
    {
        #pragma omp taskloop shared(arr,tmp) grainsize(grainsize)
        for (int i = l; i < r; i += grainsize) {
            int start = i;
            int end   = std::min(i + grainsize, r);
            merge_sort_serial(arr, tmp, start, end);
        }
    }

    for (int size = grainsize; size < n; size *= 2) {
        for (int left = l; left < r; left += 2 * size) {
            int mid   = std::min(left + size, r);
            int right = std::min(left + 2 * size, r);
            merge(arr, tmp, left, mid, right);
        }
    }
}

void obdd_parallel_var_ordering_omp(OBDD* bdd)
{
    if (!bdd || bdd->numVars <= 1) return;

    int* vo = bdd->varOrder;
    const int n = bdd->numVars;
    std::vector<int> tmp(n);
    const int task_cutoff = compute_task_cutoff();

#ifndef NDEBUG
    std::vector<int> original(vo, vo + n);
#endif

    #pragma omp parallel
    {
        #pragma omp single nowait
        merge_sort_parallel(vo, tmp.data(), 0, n, task_cutoff);
    }

#ifndef NDEBUG
    std::vector<int> check = original;
    merge_sort_serial(check.data(), tmp.data(), 0, n);
    for (int i = 0; i < n; ++i)
        assert(vo[i] == check[i]);
#endif
}

} /* extern "C" */
