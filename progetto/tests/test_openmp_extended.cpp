/**
 * @file test_openmp_extended.cpp
 * @brief Extended OpenMP backend tests for comprehensive coverage
 */

#include "core/obdd.hpp" 
#include "advanced/obdd_reordering.hpp"
#include "advanced/obdd_advanced_math.hpp"
#include <gtest/gtest.h>
#include <vector>
#include <omp.h>
#include <chrono>

#ifdef OBDD_ENABLE_OPENMP

class OpenMPExtendedTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Ensure consistent thread count
        omp_set_num_threads(4);
    }
};

TEST_F(OpenMPExtendedTest, ErrorHandling) {
    // Test NULL inputs
    EXPECT_EQ(obdd_parallel_and_omp(nullptr, nullptr), nullptr);
    EXPECT_EQ(obdd_parallel_or_omp(nullptr, nullptr), nullptr);
    EXPECT_EQ(obdd_parallel_not_omp(nullptr), nullptr);
    
    // Test with valid and NULL BDD
    int order[3] = {0,1,2};
    OBDD* bdd = obdd_create(3, order);
    bdd->root = obdd_node_create(0, obdd_constant(0), obdd_constant(1));
    
    EXPECT_EQ(obdd_parallel_and_omp(bdd, nullptr), nullptr);
    EXPECT_EQ(obdd_parallel_or_omp(bdd, nullptr), nullptr);
    
    obdd_destroy(bdd);
}

TEST_F(OpenMPExtendedTest, ComplexBDDStructures) {
    int order[8] = {0,1,2,3,4,5,6,7};
    
    // Create complex nested BDD: (x0 & x1) | (x2 & x3) | (x4 & x5) | (x6 & x7)
    OBDD* bdd1 = obdd_create(8, order);
    OBDD* bdd2 = obdd_create(8, order);
    OBDD* bdd3 = obdd_create(8, order);
    OBDD* bdd4 = obdd_create(8, order);
    
    // Create individual variable BDDs
    OBDD* var0 = obdd_create(8, order);
    OBDD* var1 = obdd_create(8, order);
    OBDD* var2 = obdd_create(8, order);
    OBDD* var3 = obdd_create(8, order);
    OBDD* var4 = obdd_create(8, order);
    OBDD* var5 = obdd_create(8, order);
    OBDD* var6 = obdd_create(8, order);
    OBDD* var7 = obdd_create(8, order);
    
    var0->root = obdd_node_create(0, obdd_constant(0), obdd_constant(1));
    var1->root = obdd_node_create(1, obdd_constant(0), obdd_constant(1));
    var2->root = obdd_node_create(2, obdd_constant(0), obdd_constant(1));
    var3->root = obdd_node_create(3, obdd_constant(0), obdd_constant(1));
    var4->root = obdd_node_create(4, obdd_constant(0), obdd_constant(1));
    var5->root = obdd_node_create(5, obdd_constant(0), obdd_constant(1));
    var6->root = obdd_node_create(6, obdd_constant(0), obdd_constant(1));
    var7->root = obdd_node_create(7, obdd_constant(0), obdd_constant(1));
    
    bdd1->root = obdd_apply(var0, var1, OBDD_AND);
    bdd2->root = obdd_apply(var2, var3, OBDD_AND);
    bdd3->root = obdd_apply(var4, var5, OBDD_AND);
    bdd4->root = obdd_apply(var6, var7, OBDD_AND);
    
    obdd_destroy(var0); obdd_destroy(var1); obdd_destroy(var2); obdd_destroy(var3);
    obdd_destroy(var4); obdd_destroy(var5); obdd_destroy(var6); obdd_destroy(var7);
    
    // Test parallel operations on complex structures
    OBDDNode* result1 = obdd_parallel_or_omp(bdd1, bdd2);
    ASSERT_NE(result1, nullptr);
    
    OBDD* intermediate = obdd_create(8, order);
    intermediate->root = result1;
    
    OBDDNode* result2 = obdd_parallel_or_omp(intermediate, bdd3);
    ASSERT_NE(result2, nullptr);
    
    obdd_destroy(bdd1);
    obdd_destroy(bdd2); 
    obdd_destroy(bdd3);
    obdd_destroy(bdd4);
    obdd_destroy(intermediate);
}

TEST_F(OpenMPExtendedTest, PerformanceScaling) {
    const int num_vars = 12;
    int order[num_vars];
    for (int i = 0; i < num_vars; i++) order[i] = i;
    
    // Create larger BDD for performance testing
    OBDD* bdd1 = obdd_create(num_vars, order);
    OBDD* bdd2 = obdd_create(num_vars, order);
    
    // Build demo BDDs
    bdd1->root = build_demo_bdd(order, num_vars);
    bdd2->root = build_demo_bdd(order, num_vars);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    OBDDNode* result = obdd_parallel_and_omp(bdd1, bdd2);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    ASSERT_NE(result, nullptr);
    EXPECT_LT(duration.count(), 100000); // Should complete in reasonable time
    
    obdd_destroy(bdd1);
    obdd_destroy(bdd2);
}

TEST_F(OpenMPExtendedTest, ThreadSafetyStress) {
    const int iterations = 100;
    const int num_threads = omp_get_max_threads();
    std::vector<bool> results(iterations * num_threads, false);
    
    int order[4] = {0,1,2,3};
    OBDD* base_bdd = obdd_create(4, order);
    base_bdd->root = obdd_node_create(0, 
        obdd_node_create(1, obdd_constant(0), obdd_constant(1)),
        obdd_node_create(2, obdd_constant(0), obdd_constant(1))
    );
    
    #pragma omp parallel for
    for (int i = 0; i < iterations * num_threads; i++) {
        OBDD* test_bdd = obdd_create(4, order);
        test_bdd->root = obdd_node_create(3, obdd_constant(0), obdd_constant(1));
        
        OBDDNode* result = obdd_parallel_and_omp(base_bdd, test_bdd);
        results[i] = (result != nullptr);
        
        obdd_destroy(test_bdd);
    }
    
    // Verify all operations succeeded
    for (bool success : results) {
        EXPECT_TRUE(success);
    }
    
    obdd_destroy(base_bdd);
}

TEST_F(OpenMPExtendedTest, AdvancedMathParallel) {
    // Test advanced math functions with OpenMP backend
    OBDD* crypto_bdd = obdd_aes_sbox();
    ASSERT_NE(crypto_bdd, nullptr);
    
    OBDD* queens_bdd = obdd_n_queens(6); // 6x6 board for reasonable test time
    ASSERT_NE(queens_bdd, nullptr);
    
    // Test parallel operations on advanced math BDDs
    OBDDNode* combined = obdd_parallel_and_omp(crypto_bdd, queens_bdd);
    EXPECT_NE(combined, nullptr);
    
    obdd_destroy(crypto_bdd);
    obdd_destroy(queens_bdd);
}

#else
TEST(OpenMPExtendedTest, DisabledBackend) {
    GTEST_SKIP() << "OpenMP backend disabled: compile with OMP=1.";
}
#endif

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}