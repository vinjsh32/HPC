#include "obdd.hpp"
#include <cstdio>

int main() {
    // ordine naturale x0..x9
    int order[10];
    for (int i = 0; i < 10; ++i) order[i] = i;

    // Crea due BDD: f1 = x0, f2 = x9
    OBDD* bdd1 = obdd_create(10, order);
    OBDD* bdd2 = obdd_create(10, order);

    OBDDNode* n1 = obdd_node_create(0, obdd_constant(0), obdd_constant(1));
    OBDDNode* n2 = obdd_node_create(9, obdd_constant(0), obdd_constant(1));
    bdd1->root = n1;
    bdd2->root = n2;

    // Valutazione su x0=1, x9=1
    int assign[10] = {0};
    assign[0] = 1;
    assign[9] = 1;

    // AND
    OBDDNode* andRoot = obdd_apply(bdd1, bdd2, OBDD_AND);
    OBDD tmpAnd{ andRoot, 10, order };
    std::printf("x0 AND x9 = %d (atteso 1)\n", obdd_evaluate(&tmpAnd, assign));

    // OR
    OBDDNode* orRoot = obdd_apply(bdd1, bdd2, OBDD_OR);
    OBDD tmpOr{ orRoot, 10, order };
    std::printf("x0 OR  x9 = %d (atteso 1)\n", obdd_evaluate(&tmpOr, assign));

    // NOT x9
    OBDDNode* notRoot = obdd_apply(bdd2, nullptr, OBDD_NOT);
    OBDD tmpNot{ notRoot, 10, order };
    std::printf("NOT x9    = %d (atteso 0)\n", obdd_evaluate(&tmpNot, assign));

    obdd_destroy(bdd1);
    obdd_destroy(bdd2);
    return 0;
}
