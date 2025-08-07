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

    /* (x0 AND x9) */
    OBDDNode* andRoot = obdd_apply(bdd1, bdd2, 0);
    int assign[10] = {1,0,0,0,0,0,0,0,0,1};
    int res = obdd_evaluate(&(OBDD){andRoot,10,order}, assign);
    ASSERT_EQ(res, 1);

    /* OR su (1,1) = 1 */
    OBDDNode* orRoot = obdd_apply(bdd1, bdd2, 1);
    res = obdd_evaluate(&(OBDD){orRoot,10,order}, assign);
    ASSERT_EQ(res, 1);

    /* NOT(x9) con x9=1 deve dare 0 */
    OBDDNode* notRoot = obdd_apply(bdd2, NULL, 2);
    res = obdd_evaluate(&(OBDD){notRoot,10,order}, assign);
    ASSERT_EQ(res, 0);

    obdd_destroy(bdd1);
    obdd_destroy(bdd2);
}

int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
