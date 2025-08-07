#include "obdd.hpp"
#include <cassert>
#include <cstdio>

int main()
{
#ifndef OBDD_ENABLE_OPENMP
    std::printf("[TEST][OMP] Backend OpenMP disabilitato: compila con OMP=1.\n");
    return 0;
#else
    int order[2] = {0,1};
    OBDD* bddX0 = obdd_create(2, order);
    OBDD* bddX1 = obdd_create(2, order);
    bddX0->root = obdd_node_create(0, OBDD_FALSE, OBDD_TRUE);
    bddX1->root = obdd_node_create(1, OBDD_FALSE, OBDD_TRUE);

    OBDDNode* andRoot = obdd_parallel_and_omp(bddX0, bddX1);
    OBDDNode* orRoot  = obdd_parallel_or_omp (bddX0, bddX1);
    OBDDNode* notRoot = obdd_parallel_not_omp(bddX1);

    int assignTT[2] = {1,1};
    int assignTF[2] = {1,0};

    OBDD tmp{andRoot,2,order};
    assert(obdd_evaluate(&tmp, assignTT) == 1);
    assert(obdd_evaluate(&tmp, assignTF) == 0);
    tmp.root = orRoot;
    assert(obdd_evaluate(&tmp, assignTF) == 1);
    tmp.root = notRoot;
    assert(obdd_evaluate(&tmp, assignTF) == 1);

    int v[8] = {7,3,5,0,2,6,1,4};
    OBDD dummy{nullptr,8,v};
    obdd_parallel_var_ordering_omp(&dummy);
    for (int i = 1; i < 8; ++i)
        assert(v[i-1] <= v[i]);

    obdd_destroy(bddX0);
    obdd_destroy(bddX1);
    std::puts("[TEST][OMP] Completato con successo.");
    return 0;
#endif
}

