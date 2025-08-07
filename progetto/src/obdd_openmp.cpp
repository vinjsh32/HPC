/**
 * @file obdd_openmp.cpp
 * @brief Backend parallelo OpenMP per OBDD/ROBDD.
 *
 *  – Versioni multi-thread di apply / AND-OR-XOR-NOT e var-ordering.
 *  – Condivide la apply-cache con il core, protetta da lock.
 *  – API con linkage C (vedi obdd.hpp).
 */

#include "obdd.hpp"
#include "apply_cache.hpp"     /* memoization helpers */

#include <omp.h>
#include <climits>

/* --------------------------------------------------------------------------
 *  Ricorsione parallela con task
 * -------------------------------------------------------------------------- */
static OBDDNode* obdd_parallel_apply_internal(const OBDDNode* n1,
                                              const OBDDNode* n2,
                                              OBDD_Op         op)
{
    /* 1) memo-lookup thread-safe */
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

    #pragma omp task shared(lowRes)  firstprivate(n1_low, n2_low, op)
    lowRes  = obdd_parallel_apply_internal(n1_low,  n2_low,  op);

    #pragma omp task shared(highRes) firstprivate(n1_high, n2_high, op)
    highRes = obdd_parallel_apply_internal(n1_high, n2_high, op);

    #pragma omp taskwait

    /* 5) riduzione locale */
    OBDDNode* res = (lowRes == highRes)
                    ? lowRes
                    : obdd_node_create(var, lowRes, highRes);

    /* 6) salva in cache */
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

    #pragma omp parallel
    {
        #pragma omp single nowait
        out = obdd_parallel_apply_internal(bdd1->root,
                                           bdd2 ? bdd2->root : nullptr,
                                           op);
    }
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
 *  Odd–Even Transposition Sort sul vettore varOrder
 * ------------------------------------------------------------------ */
void obdd_parallel_var_ordering_omp(OBDD* bdd)
{
    if (!bdd || bdd->numVars <= 1) return;

    int* vo = bdd->varOrder;
    const int n = bdd->numVars;

    for (int pass = 0; pass < n; ++pass) {
        int phase = pass & 1;                        /* 0 = even, 1 = odd */
        #pragma omp parallel for default(none) shared(vo,n,phase)
        for (int i = phase; i < n - 1; i += 2) {
            if (vo[i] > vo[i + 1]) {
                int tmp = vo[i]; vo[i] = vo[i + 1]; vo[i + 1] = tmp;
            }
        }
    }
}

} /* extern "C" */
