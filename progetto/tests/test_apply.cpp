/* ---------------------------------------------------------
 *  test_apply.cpp – piccoli unit‑test con GoogleTest
 * --------------------------------------------------------- */

#include "obdd.hpp"
#include <gtest/gtest.h>

TEST(ApplyBasic, AndOrNot)
{
    int order[10];
    for (int i = 0; i < 10; ++i) order[i] = i;

    OBDD* bdd1 = obdd_create(10, order); /* x0 */
    bdd1->root = obdd_node_create(0, obdd_constant(0), obdd_constant(1));

    OBDD* bdd2 = obdd_create(10, order); /* x9 */
    bdd2->root = obdd_node_create(9, obdd_constant(0), obdd_constant(1));

    OBDD tmp = {nullptr,10,order};

    /* (x0 AND x9) */
    OBDDNode* andRoot = obdd_apply(bdd1, bdd2, OBDD_AND);
    int assign[10] = {1,0,0,0,0,0,0,0,0,1};
    tmp.root = andRoot;
    int res = obdd_evaluate(&tmp, assign);
    ASSERT_EQ(res, 1);

    /* OR su (1,1) = 1 */
    OBDDNode* orRoot = obdd_apply(bdd1, bdd2, OBDD_OR);
    tmp.root = orRoot;
    res = obdd_evaluate(&tmp, assign);
    ASSERT_EQ(res, 1);

    /* XOR su (1,0) = 1 */
    assign[0] = 1;
    assign[9] = 0;
    OBDDNode* xorRoot = obdd_apply(bdd1, bdd2, OBDD_XOR);
    tmp.root = xorRoot;
    res = obdd_evaluate(&tmp, assign);
    ASSERT_EQ(res, 1);
    assign[9] = 1; /* ripristina x9 per i test successivi */

    /* NOT(x9) con x9=1 deve dare 0 */
    OBDDNode* notRoot = obdd_apply(bdd2, NULL, OBDD_NOT);
    tmp.root = notRoot;
    res = obdd_evaluate(&tmp, assign);
    ASSERT_EQ(res, 0);

    obdd_destroy(bdd1);
    obdd_destroy(bdd2);
}

int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
