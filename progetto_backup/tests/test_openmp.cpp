#include "obdd.hpp"
#include "apply_cache.hpp"
#include <gtest/gtest.h>

OBDDNode* obdd_parallel_apply_omp_opt(const OBDD*, const OBDD*, OBDD_Op);

#ifdef OBDD_ENABLE_OPENMP

static OBDDNode* build_and_chain(int idx, int numVars)
{
    if (idx == numVars - 1)
        return obdd_node_create(idx, OBDD_FALSE, OBDD_TRUE);
    return obdd_node_create(idx, OBDD_FALSE, build_and_chain(idx + 1, numVars));
}

static OBDDNode* build_or_chain(int idx, int numVars)
{
    if (idx == numVars - 1)
        return obdd_node_create(idx, OBDD_FALSE, OBDD_TRUE);
    return obdd_node_create(idx, build_or_chain(idx + 1, numVars), OBDD_TRUE);
}

TEST(OpenMPBackend, LogicalOperations)
{
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
    EXPECT_EQ(obdd_evaluate(&tmp, assignTT), 1);
    EXPECT_EQ(obdd_evaluate(&tmp, assignTF), 0);
    tmp.root = orRoot;
    EXPECT_EQ(obdd_evaluate(&tmp, assignTF), 1);
    tmp.root = notRoot;
    EXPECT_EQ(obdd_evaluate(&tmp, assignTF), 1);

    obdd_destroy(bddX0);
    obdd_destroy(bddX1);
}

TEST(OpenMPBackend, OptimizedMatchesSequential)
{
    int order[2] = {0,1};
    OBDD* bddX0 = obdd_create(2, order);
    OBDD* bddX1 = obdd_create(2, order);
    bddX0->root = obdd_node_create(0, OBDD_FALSE, OBDD_TRUE);
    bddX1->root = obdd_node_create(1, OBDD_FALSE, OBDD_TRUE);

    int inputs[4][2] = {{0,0},{0,1},{1,0},{1,1}};

    OBDDNode* seq_and = obdd_apply(bddX0, bddX1, OBDD_AND);
    OBDDNode* par_and = obdd_parallel_apply_omp_opt(bddX0, bddX1, OBDD_AND);
    OBDD seqBDD{seq_and,2,order};
    OBDD parBDD{par_and,2,order};
    for (auto& in : inputs)
        EXPECT_EQ(obdd_evaluate(&seqBDD, in), obdd_evaluate(&parBDD, in));

    seqBDD.root = obdd_apply(bddX0, bddX1, OBDD_OR);
    parBDD.root = obdd_parallel_apply_omp_opt(bddX0, bddX1, OBDD_OR);
    for (auto& in : inputs)
        EXPECT_EQ(obdd_evaluate(&seqBDD, in), obdd_evaluate(&parBDD, in));

    seqBDD.root = obdd_apply(bddX0, bddX1, OBDD_XOR);
    parBDD.root = obdd_parallel_apply_omp_opt(bddX0, bddX1, OBDD_XOR);
    for (auto& in : inputs)
        EXPECT_EQ(obdd_evaluate(&seqBDD, in), obdd_evaluate(&parBDD, in));

    seqBDD.root = obdd_apply(bddX1, nullptr, OBDD_NOT);
    parBDD.root = obdd_parallel_apply_omp_opt(bddX1, nullptr, OBDD_NOT);
    for (auto& in : inputs)
        EXPECT_EQ(obdd_evaluate(&seqBDD, in), obdd_evaluate(&parBDD, in));

    obdd_destroy(bddX0);
    obdd_destroy(bddX1);
}

TEST(OpenMPBackend, ParallelXorLargeBDD)
{
    constexpr int N = 10;
    int order[N];
    for (int i = 0; i < N; ++i) order[i] = i;

    OBDD* bddA = obdd_create(N, order);
    OBDD* bddB = obdd_create(N, order);
    bddA->root = build_and_chain(0, N);
    bddB->root = build_or_chain(0, N);

    OBDDNode* seqRoot = obdd_apply(bddA, bddB, OBDD_XOR);
    OBDDNode* parRoot = obdd_parallel_xor_omp(bddA, bddB);
    OBDD seqBDD{seqRoot, N, order};
    OBDD parBDD{parRoot, N, order};

    int inputs[N];
    for (int mask = 0; mask < (1 << N); ++mask) {
        for (int j = 0; j < N; ++j)
            inputs[j] = (mask >> j) & 1;
        EXPECT_EQ(obdd_evaluate(&seqBDD, inputs),
                  obdd_evaluate(&parBDD, inputs));
    }

    obdd_destroy(bddA);
    obdd_destroy(bddB);
}

TEST(OpenMPBackend, ParallelApplyLargeBDD)
{
    constexpr int N = 10;
    int order[N];
    for (int i = 0; i < N; ++i) order[i] = i;

    OBDD* bddA = obdd_create(N, order);
    OBDD* bddB = obdd_create(N, order);
    bddA->root = build_and_chain(0, N);
    bddB->root = build_or_chain(0, N);

    OBDDNode* seqRoot = obdd_apply(bddA, bddB, OBDD_AND);
    OBDDNode* parRoot = obdd_parallel_apply_omp(bddA, bddB, OBDD_AND);
    OBDD seqBDD{seqRoot, N, order};
    OBDD parBDD{parRoot, N, order};

    int inputs[N];
    for (int mask = 0; mask < (1 << N); ++mask) {
        for (int j = 0; j < N; ++j)
            inputs[j] = (mask >> j) & 1;
        EXPECT_EQ(obdd_evaluate(&seqBDD, inputs),
                  obdd_evaluate(&parBDD, inputs));
    }

    obdd_destroy(bddA);
    obdd_destroy(bddB);
}

TEST(OpenMPBackend, VarOrdering)
{
    int v[8] = {7,3,5,0,2,6,1,4};
    OBDD dummy{nullptr,8,v};
    obdd_parallel_var_ordering_omp(&dummy);
    for (int i = 1; i < 8; ++i)
        EXPECT_LE(v[i-1], v[i]);
}

TEST(OpenMPBackend, VarOrderingComplexVector)
{
    int v[12] = {7,3,5,0,2,6,1,4,4,2,3,7};
    OBDD dummy{nullptr,12,v};
    obdd_parallel_var_ordering_omp(&dummy);
    for (int i = 1; i < 12; ++i)
        EXPECT_LE(v[i-1], v[i]);
}

TEST(OpenMPBackend, SelfAndShortCircuit)
{
    int order[1] = {0};
    OBDD* bdd = obdd_create(1, order);
    bdd->root = obdd_node_create(0, OBDD_FALSE, OBDD_TRUE);

    OBDDNode* res = obdd_parallel_and_omp(bdd, bdd);
    OBDD tmp{res,1,order};
    int in[1] = {0};
    EXPECT_EQ(obdd_evaluate(&tmp, in), obdd_evaluate(bdd, in));
    in[0] = 1;
    EXPECT_EQ(obdd_evaluate(&tmp, in), obdd_evaluate(bdd, in));

    obdd_destroy(bdd);
}

TEST(OpenMPBackend, NullInput)
{
    int order[1] = {0};
    OBDD* bdd = obdd_create(1, order);
    bdd->root = obdd_node_create(0, OBDD_FALSE, OBDD_TRUE);

    EXPECT_EQ(obdd_parallel_and_omp(nullptr, bdd), nullptr);
    EXPECT_EQ(obdd_parallel_and_omp(bdd, nullptr), nullptr);
    EXPECT_EQ(obdd_parallel_not_omp(nullptr), nullptr);

    obdd_destroy(bdd);
}

TEST(OpenMPBackend, VarOrderingNoChange)
{
    int v[8] = {0,1,1,2,3,3,4,4};
    OBDD dummy{nullptr,8,v};
    obdd_parallel_var_ordering_omp(&dummy);
    int expected[8] = {0,1,1,2,3,3,4,4};
    for (int i = 0; i < 8; ++i)
        EXPECT_EQ(v[i], expected[i]);
    obdd_parallel_var_ordering_omp(nullptr);
}

#else

TEST(OpenMPBackend, DisabledBackend)
{
    GTEST_SKIP() << "Backend OpenMP disabilitato: compila con OMP=1.";
}

#endif

int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

