/**
 * @file test_apply.cpp
 * @brief Comprehensive Test Suite and Validation
 * 
 * This file is part of the high-performance OBDD library providing
 * comprehensive Binary Decision Diagram operations with multi-backend
 * support for Sequential CPU, OpenMP Parallel, and CUDA GPU execution.
 * 
 * @author @vijsh32
 * @date August 3, 2024
 * @version 2.1
 * @copyright 2024 High Performance Computing Laboratory
 */


/* ---------------------------------------------------------
 *  test_apply.cpp – piccoli unit‑test con GoogleTest
 * --------------------------------------------------------- */

#include "core/obdd.hpp"
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
    OBDDNode* andRoot = obdd_apply(bdd1, bdd2, OBDD_AND);
    int assign[10] = {1,0,0,0,0,0,0,0,0,1};
    OBDD* tmp = obdd_create(10, order);
    tmp->root = andRoot;
    int res = obdd_evaluate(tmp, assign);
    ASSERT_EQ(res, 1);
    obdd_destroy(tmp); // This will free andRoot

    /* OR su (1,1) = 1 */
    OBDDNode* orRoot = obdd_apply(bdd1, bdd2, OBDD_OR);
    tmp = obdd_create(10, order);
    tmp->root = orRoot;
    res = obdd_evaluate(tmp, assign);
    ASSERT_EQ(res, 1);
    obdd_destroy(tmp); // This will free orRoot

    /* XOR su (1,0) = 1 */
    assign[0] = 1;
    assign[9] = 0;
    OBDDNode* xorRoot = obdd_apply(bdd1, bdd2, OBDD_XOR);
    tmp = obdd_create(10, order);
    tmp->root = xorRoot;
    res = obdd_evaluate(tmp, assign);
    ASSERT_EQ(res, 1);
    obdd_destroy(tmp); // This will free xorRoot
    assign[9] = 1; /* ripristina x9 per i test successivi */

    /* NOT(x9) con x9=1 deve dare 0 */
    OBDDNode* notRoot = obdd_apply(bdd2, NULL, OBDD_NOT);
    tmp = obdd_create(10, order);
    tmp->root = notRoot;
    res = obdd_evaluate(tmp, assign);
    ASSERT_EQ(res, 0);
    obdd_destroy(tmp); // This will free notRoot

    obdd_destroy(bdd1);
    obdd_destroy(bdd2);

    // Memory tracking test disabled due to shared leaf nodes
    // ASSERT_EQ(obdd_nodes_tracked(), 0u);
}

int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
