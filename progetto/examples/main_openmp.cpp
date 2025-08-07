/**
 *  main_openmp.cpp
 *  ----------------
 *  Collaudo rapido del backend OpenMP:
 *    – costruzione di due BDD di prova
 *    – var-ordering parallelo
 *    – AND / OR / XOR / NOT paralleli
 *    – valutazione di un assegnamento
 *
 *  Compilazione (esempio GCC):
 *      g++  -std=c++17  -fopenmp  -O2  \
 *          obdd_core.cpp  obdd_openmp.cpp  main_openmp.cpp  -o test_omp
 */

#include "obdd.hpp"
#include <cstdio>
#include <cstring>
#include <omp.h>
#include "obdd.hpp"

/* ----------------------------------------------------------
 *  utility: conta nodi diversi con una DFS (no ricorsione)
 * ---------------------------------------------------------- */
static int obdd_size(const OBDDNode* root)
{
    if (!root) return 0;
    const int MAX = 100000;
    const OBDDNode* visited[MAX];
    int top = 0, count = 0;
    visited[top++] = root;

    for (int i = 0; i < top; ++i) {
        const OBDDNode* cur = visited[i];
        ++count;
        if (cur->varIndex < 0) continue;          // leaf
        /* push figli se non già visti */
        bool seenLow = false, seenHigh = false;
        for (int j = 0; j < top; ++j) {
            seenLow  |= (visited[j] == cur->lowChild);
            seenHigh |= (visited[j] == cur->highChild);
        }
        if (!seenLow  && top < MAX) visited[top++] = cur->lowChild;
        if (!seenHigh && top < MAX) visited[top++] = cur->highChild;
    }
    return count;
}

int main()
{
    omp_set_num_threads(4);

    /* -------  varOrder iniziale 0..9 (verrà permutato) -------- */
    int order[10]; for (int i = 0; i < 10; ++i) order[i] = i;

    /* -------  BDD demo:  f = (x0∧x1) ∨ (x2⊕x4)  ---------------- */
    OBDD* bdd = obdd_create(10, order);
    bdd->root = build_demo_bdd(bdd->varOrder, bdd->numVars);

    printf("[OMP] size(demo) prima del sort  = %d\n", obdd_size(bdd->root));

    /* -------  Parallel var-ordering (bubble sort) --------------- */
    obdd_parallel_var_ordering_omp(bdd);
    printf("[OMP] varOrder dopo bubble-sort  = ");
    for (int i = 0; i < 10; ++i) printf("%d ", bdd->varOrder[i]);
    puts("");

    /* -------  BDD2 =  (x0 ∨ x9)  ------------------------------- */
    OBDD* bdd2 = obdd_create(10, order);
    int idx0 = 0, idx9 = 9;               /* dopo sort l’ordine è 0..9 */
    OBDDNode* x0 = obdd_node_create(idx0, obdd_constant(0), obdd_constant(1));
    OBDDNode* x9 = obdd_node_create(idx9, obdd_constant(0), obdd_constant(1));
    OBDD tmpA{ x0, 10, bdd2->varOrder };
    OBDD tmpB{ x9, 10, bdd2->varOrder };
    bdd2->root = obdd_parallel_or_omp(&tmpA, &tmpB);   /* OR seriale o parallelo */

    /* -------  Operazioni parallele ----------------------------- */
    OBDDNode* andRes = obdd_parallel_and_omp(bdd, bdd2);
    OBDDNode* orRes  = obdd_parallel_or_omp (bdd, bdd2);
    OBDDNode* xorRes = obdd_parallel_xor_omp(bdd, bdd2);
    OBDDNode* notRes = obdd_parallel_not_omp(bdd2);

    printf("[OMP] size(AND) = %d\n", obdd_size(andRes));
    printf("[OMP] size(OR)  = %d\n", obdd_size(orRes));
    printf("[OMP] size(XOR) = %d\n", obdd_size(xorRes));
    printf("[OMP] size(NOT) = %d\n", obdd_size(notRes));

    /* -------  Una valutazione di prova -------------------------- */
    int assign[10]; std::memset(assign, 0, sizeof(assign));
    assign[0] = assign[1] = assign[2] = assign[9] = 1;     // x0=x1=x2=x9=1

    OBDD fakeAnd{ andRes, 10, bdd->varOrder };
    printf("[OMP] AND(assign) = %d  (atteso 1)\n",
           obdd_evaluate(&fakeAnd, assign));

    /* -------  cleanup ------------------------------------------ */
    obdd_destroy(bdd);
    obdd_destroy(bdd2);
    return 0;
}
