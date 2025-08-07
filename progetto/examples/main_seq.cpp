/* ---------------------------------------------------------
 *  main_seq.cpp – demo sequenziale di 10 variabili
 *  Compila e linka contro la libreria obdd_core
 * --------------------------------------------------------- */

#include <stdio.h>
#include "obdd.hpp"

int main()
{
    /* ordine naturale x0..x9 */
    int order[10];
    for (int i = 0; i < 10; ++i) order[i] = i;

    /* Crea due BDD: f1 = x0, f2 = x9 */
    OBDD* bdd1 = obdd_create(10, order);
    OBDD* bdd2 = obdd_create(10, order);

    OBDDNode* n1 = obdd_node_create(0, obdd_constant(0), obdd_constant(1));
    OBDDNode* n2 = obdd_node_create(9, obdd_constant(0), obdd_constant(1));
    bdd1->root = n1;
    bdd2->root = n2;

    /* Valuta x0 con x0=0 */
    int assign1[10] = {0}; /* tutto 0 */
    int res = obdd_evaluate(bdd1, assign1);
    printf("x0(0) = %d (atteso 0)\n", res);

    /* x0=1, x9=1 */
    int assign2[10] = {0}; assign2[0]=1; assign2[9]=1;

    /* AND */
    OBDDNode* andRoot = obdd_apply(bdd1, bdd2, 0);
    OBDD tmpAnd { andRoot, 10, order };
    printf("x0 AND x9 = %d (atteso 1)\n", obdd_evaluate(&tmpAnd, assign2));

    /* OR */
    OBDDNode* orRoot = obdd_apply(bdd1, bdd2, 1);
    OBDD tmpOr { orRoot, 10, order };
    printf("x0 OR  x9 = %d (atteso 1)\n", obdd_evaluate(&tmpOr, assign2));

    /* NOT x9 */
    OBDDNode* notRoot = obdd_apply(bdd2, nullptr, 2);
    OBDD tmpNot { notRoot, 10, order };
    printf("NOT x9    = %d (atteso 0)\n", obdd_evaluate(&tmpNot, assign2));

    obdd_destroy(bdd1);
    obdd_destroy(bdd2);
    return 0;
}
