/**
 * @file test_reordering.cpp  
 * @brief Comprehensive tests for advanced variable reordering algorithms
 */

#include "advanced/obdd_reordering.hpp"
#include "core/obdd.hpp"
#include <gtest/gtest.h>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <iostream>

class ReorderingTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create test BDD representing a Boolean function
        // f(x0,x1,x2,x3) = (x0 ∧ x1) ∨ (x2 ∧ x3)
        int order[4] = {0, 1, 2, 3};
        test_bdd = obdd_create(4, order);
        
        // Build BDD structure: (x0 ∧ x1) ∨ (x2 ∧ x3)
        OBDDNode* x0_and_x1 = obdd_node_create(0, 
            obdd_node_create(1, OBDD_FALSE, OBDD_FALSE), // x0=0: x1 doesn't matter -> false
            obdd_node_create(1, OBDD_FALSE, OBDD_TRUE)    // x0=1: result = x1
        );
        
        OBDDNode* x2_and_x3 = obdd_node_create(2,
            obdd_node_create(3, OBDD_FALSE, OBDD_FALSE), // x2=0: x3 doesn't matter -> false  
            obdd_node_create(3, OBDD_FALSE, OBDD_TRUE)    // x2=1: result = x3
        );
        
        // Combine with OR
        test_bdd->root = obdd_node_create(0,
            obdd_node_create(1,
                x2_and_x3,  // x0=0,x1=0: result = x2∧x3
                x2_and_x3   // x0=0,x1=1: result = x2∧x3
            ),
            obdd_node_create(1,
                x2_and_x3,  // x0=1,x1=0: result = x2∧x3
                OBDD_TRUE   // x0=1,x1=1: result = true
            )
        );
        
        // Reduce to canonical form
        test_bdd->root = obdd_reduce(test_bdd->root);
        
        // Create larger BDD for stress testing
        int large_order[8] = {0, 1, 2, 3, 4, 5, 6, 7};
        large_bdd = obdd_create(8, large_order);
        
        // Build chain: x0 → x1 → x2 → ... → x7 → TRUE
        OBDDNode* chain = OBDD_TRUE;
        for (int i = 7; i >= 0; i--) {
            chain = obdd_node_create(i, OBDD_FALSE, chain);
        }
        large_bdd->root = obdd_reduce(chain);
    }
    
    void TearDown() override {
        obdd_destroy(test_bdd);
        obdd_destroy(large_bdd);
    }
    
    // Helper function to build a multiplier BDD for advanced testing
    OBDD* build_multiplier_bdd(int bits) {
        int num_vars = bits * 3; // a_bits + b_bits + product_bits  
        int* order = new int[num_vars];
        std::iota(order, order + num_vars, 0);
        
        OBDD* mult_bdd = obdd_create(num_vars, order);
        
        // Simplified 2-bit multiplier: a1a0 * b1b0 = p3p2p1p0
        // This is a placeholder - full multiplier would be much more complex
        if (bits == 2) {
            // Variables: a1(0), a0(1), b1(2), b0(3), p3(4), p2(5), p1(6), p0(7)
            // Constraint: p0 = a0 ∧ b0
            mult_bdd->root = obdd_node_create(1, // a0
                obdd_node_create(3, OBDD_FALSE, // a0=0,b0=?: p0=0
                                 obdd_node_create(6, OBDD_TRUE, OBDD_FALSE)), // p0=0 if a0=0
                obdd_node_create(3, // a0=1,b0=?
                                 obdd_node_create(6, OBDD_TRUE, OBDD_FALSE), // b0=0: p0=0
                                 obdd_node_create(6, OBDD_FALSE, OBDD_TRUE))  // b0=1: p0=1
            );
        } else {
            // For larger multipliers, use a simpler pattern
            mult_bdd->root = obdd_node_create(0, OBDD_FALSE, OBDD_TRUE);
        }
        
        mult_bdd->root = obdd_reduce(mult_bdd->root);
        delete[] order;
        return mult_bdd;
    }
    
    OBDD* test_bdd;
    OBDD* large_bdd;
};

TEST_F(ReorderingTest, DefaultConfigurations) {
    ReorderConfig sifting_config = obdd_reorder_get_default_config(REORDER_SIFTING);
    EXPECT_EQ(sifting_config.strategy, REORDER_SIFTING);
    EXPECT_EQ(sifting_config.max_iterations, 10);
    EXPECT_TRUE(sifting_config.enable_parallel);
    
    ReorderConfig window_config = obdd_reorder_get_default_config(REORDER_WINDOW_DP);
    EXPECT_EQ(window_config.strategy, REORDER_WINDOW_DP);
    EXPECT_EQ(window_config.window_size, 4);
    EXPECT_FALSE(window_config.enable_parallel);
    
    ReorderConfig sa_config = obdd_reorder_get_default_config(REORDER_SIMULATED_ANNEALING);
    EXPECT_EQ(sa_config.strategy, REORDER_SIMULATED_ANNEALING);
    EXPECT_EQ(sa_config.max_iterations, 1000);
    EXPECT_DOUBLE_EQ(sa_config.temperature_initial, 100.0);
    EXPECT_DOUBLE_EQ(sa_config.cooling_rate, 0.95);
    
    ReorderConfig genetic_config = obdd_reorder_get_default_config(REORDER_GENETIC);
    EXPECT_EQ(genetic_config.strategy, REORDER_GENETIC);
    EXPECT_EQ(genetic_config.population_size, 100);
    EXPECT_DOUBLE_EQ(genetic_config.mutation_rate, 0.02);
    EXPECT_DOUBLE_EQ(genetic_config.crossover_rate, 0.8);
}

TEST_F(ReorderingTest, SiftingAlgorithm) {
    ReorderResult result = {};
    int initial_size = obdd_count_nodes(test_bdd);
    
    int* new_order = obdd_reorder_sifting(test_bdd, 5, obdd_count_nodes, &result);
    
    ASSERT_NE(new_order, nullptr);
    EXPECT_EQ(result.initial_size, initial_size);
    EXPECT_LE(result.final_size, initial_size); // Should not increase size
    EXPECT_GE(result.reduction_ratio, 0.0);
    EXPECT_LE(result.reduction_ratio, 1.0);
    EXPECT_GT(result.execution_time_ms, 0.0);
    EXPECT_STREQ(result.algorithm_used, "Sifting");
    
    std::cout << "Sifting Results:" << std::endl;
    obdd_print_reorder_result(&result);
    
    // Verify ordering is valid permutation
    std::vector<int> order_vec(new_order, new_order + test_bdd->numVars);
    std::sort(order_vec.begin(), order_vec.end());
    for (int i = 0; i < test_bdd->numVars; i++) {
        EXPECT_EQ(order_vec[i], i);
    }
    
    std::free(new_order);
}

TEST_F(ReorderingTest, WindowPermutationDP) {
    ReorderResult result = {};
    int initial_size = obdd_count_nodes(test_bdd);
    
    int* new_order = obdd_reorder_window_dp(test_bdd, 3, obdd_count_nodes, &result);
    
    ASSERT_NE(new_order, nullptr);
    EXPECT_EQ(result.initial_size, initial_size);
    EXPECT_LE(result.final_size, initial_size);
    EXPECT_STREQ(result.algorithm_used, "Window DP");
    
    std::cout << "Window DP Results:" << std::endl;
    obdd_print_reorder_result(&result);
    
    // Verify valid permutation
    std::vector<int> order_vec(new_order, new_order + test_bdd->numVars);
    std::sort(order_vec.begin(), order_vec.end());
    for (int i = 0; i < test_bdd->numVars; i++) {
        EXPECT_EQ(order_vec[i], i);
    }
    
    std::free(new_order);
}

TEST_F(ReorderingTest, SimulatedAnnealing) {
    ReorderResult result = {};
    int initial_size = obdd_count_nodes(test_bdd);
    
    int* new_order = obdd_reorder_simulated_annealing(test_bdd, 
                                                     50.0,  // initial temp
                                                     0.9,   // cooling rate  
                                                     100,   // max iterations
                                                     obdd_count_nodes, 
                                                     &result);
    
    ASSERT_NE(new_order, nullptr);
    EXPECT_EQ(result.initial_size, initial_size);
    EXPECT_LE(result.final_size, initial_size);
    EXPECT_EQ(result.iterations_performed, 100);
    EXPECT_STREQ(result.algorithm_used, "Simulated Annealing");
    
    std::cout << "Simulated Annealing Results:" << std::endl;
    obdd_print_reorder_result(&result);
    
    std::free(new_order);
}

TEST_F(ReorderingTest, GeneticAlgorithm) {
    ReorderResult result = {};
    int initial_size = obdd_count_nodes(test_bdd);
    
    int* new_order = obdd_reorder_genetic(test_bdd,
                                         20,   // population size
                                         10,   // generations
                                         0.1,  // mutation rate
                                         0.8,  // crossover rate
                                         obdd_count_nodes,
                                         &result);
    
    ASSERT_NE(new_order, nullptr);
    EXPECT_EQ(result.initial_size, initial_size);
    EXPECT_LE(result.final_size, initial_size);
    EXPECT_EQ(result.iterations_performed, 10);
    EXPECT_STREQ(result.algorithm_used, "Genetic Algorithm");
    
    std::cout << "Genetic Algorithm Results:" << std::endl;
    obdd_print_reorder_result(&result);
    
    std::free(new_order);
}

TEST_F(ReorderingTest, MainReorderingAPI) {
    // Test unified API with different strategies
    std::vector<ReorderStrategy> strategies = {
        REORDER_SIFTING,
        REORDER_WINDOW_DP,
        REORDER_SIMULATED_ANNEALING,
        REORDER_GENETIC,
        REORDER_HYBRID
    };
    
    int initial_size = obdd_count_nodes(test_bdd);
    
    for (ReorderStrategy strategy : strategies) {
        // Create a copy of BDD for each test
        int order[4] = {0, 1, 2, 3};
        OBDD* bdd_copy = obdd_create(4, order);
        bdd_copy->root = test_bdd->root;
        
        ReorderConfig config = obdd_reorder_get_default_config(strategy);
        if (strategy == REORDER_WINDOW_DP) {
            config.window_size = 2; // Smaller window for speed
        }
        if (strategy == REORDER_GENETIC) {
            config.population_size = 10;
            config.max_iterations = 5;
        }
        if (strategy == REORDER_SIMULATED_ANNEALING || strategy == REORDER_HYBRID) {
            config.max_iterations = 50;
        }
        
        ReorderResult result = {};
        int* new_order = obdd_reorder_advanced(bdd_copy, &config, &result);
        
        ASSERT_NE(new_order, nullptr) << "Strategy " << static_cast<int>(strategy) << " failed";
        EXPECT_EQ(result.initial_size, initial_size);
        EXPECT_LE(result.final_size, initial_size);
        EXPECT_GT(result.execution_time_ms, 0.0);
        
        std::cout << "Strategy " << static_cast<int>(strategy) << " Results:" << std::endl;
        obdd_print_reorder_result(&result);
        
        std::free(new_order);
        obdd_destroy(bdd_copy);
    }
}

TEST_F(ReorderingTest, LargeScalePerformance) {
    // Test on larger BDD to see performance characteristics
    int initial_size = obdd_count_nodes(large_bdd);
    
    ReorderConfig config = obdd_reorder_get_default_config(REORDER_SIFTING);
    config.max_iterations = 3; // Reduced for speed
    
    ReorderResult result = {};
    int* new_order = obdd_reorder_advanced(large_bdd, &config, &result);
    
    ASSERT_NE(new_order, nullptr);
    EXPECT_GT(result.execution_time_ms, 0.0);
    EXPECT_EQ(result.initial_size, initial_size);
    
    std::cout << "Large Scale Sifting Results:" << std::endl;
    obdd_print_reorder_result(&result);
    
    std::free(new_order);
}

TEST_F(ReorderingTest, MultiplierOptimization) {
    // Test on a more realistic multiplier BDD
    OBDD* mult_bdd = build_multiplier_bdd(2);
    int initial_size = obdd_count_nodes(mult_bdd);
    
    std::cout << "Testing reordering on 2-bit multiplier BDD (initial size: " 
              << initial_size << " nodes)" << std::endl;
    
    // Try different algorithms
    std::vector<ReorderStrategy> strategies = {REORDER_SIFTING, REORDER_GENETIC};
    
    for (ReorderStrategy strategy : strategies) {
        int order[8] = {0, 1, 2, 3, 4, 5, 6, 7};
        OBDD* mult_copy = obdd_create(8, order);
        mult_copy->root = mult_bdd->root;
        
        ReorderConfig config = obdd_reorder_get_default_config(strategy);
        if (strategy == REORDER_GENETIC) {
            config.population_size = 20;
            config.max_iterations = 10;
        }
        
        ReorderResult result = {};
        int* new_order = obdd_reorder_advanced(mult_copy, &config, &result);
        
        ASSERT_NE(new_order, nullptr);
        
        std::cout << "Multiplier - Strategy " << static_cast<int>(strategy) << ":" << std::endl;
        obdd_print_reorder_result(&result);
        
        std::free(new_order);
        obdd_destroy(mult_copy);
    }
    
    obdd_destroy(mult_bdd);
}

#ifdef OBDD_ENABLE_OPENMP
TEST_F(ReorderingTest, ParallelAlgorithms) {
    ReorderResult sifting_result = {};
    int* sifting_order = obdd_reorder_sifting_omp(test_bdd, 3, obdd_count_nodes, &sifting_result);
    
    ASSERT_NE(sifting_order, nullptr);
    EXPECT_STREQ(sifting_result.algorithm_used, "Sifting OpenMP");
    EXPECT_GT(sifting_result.execution_time_ms, 0.0);
    
    std::cout << "Parallel Sifting Results:" << std::endl;
    obdd_print_reorder_result(&sifting_result);
    
    std::free(sifting_order);
    
    // Test parallel genetic algorithm
    ReorderResult genetic_result = {};
    int* genetic_order = obdd_reorder_genetic_omp(test_bdd, 10, 5, 0.1, 0.8, 
                                                  obdd_count_nodes, &genetic_result);
    
    ASSERT_NE(genetic_order, nullptr);
    EXPECT_STREQ(genetic_result.algorithm_used, "Genetic Algorithm OpenMP");
    
    std::cout << "Parallel Genetic Results:" << std::endl;
    obdd_print_reorder_result(&genetic_result);
    
    std::free(genetic_order);
}
#endif

TEST_F(ReorderingTest, UtilityFunctions) {
    // Test node counting
    int node_count = obdd_count_nodes(test_bdd);
    EXPECT_GT(node_count, 0);
    
    int memory_footprint = obdd_count_memory_footprint(test_bdd);
    EXPECT_EQ(memory_footprint, node_count * sizeof(OBDDNode));
    
    // Test random ordering generation
    int* random_order = obdd_generate_random_ordering(4, 12345);
    ASSERT_NE(random_order, nullptr);
    
    std::vector<int> order_check(random_order, random_order + 4);
    std::sort(order_check.begin(), order_check.end());
    for (int i = 0; i < 4; i++) {
        EXPECT_EQ(order_check[i], i);
    }
    
    std::free(random_order);
    
    // Test variable swapping
    int original_size = obdd_count_nodes(test_bdd);
    int new_size = obdd_swap_adjacent_variables(test_bdd, 1);
    EXPECT_GT(new_size, 0);
    // Size might increase or decrease depending on the function
}

TEST_F(ReorderingTest, AlgorithmComparison) {
    std::cout << "\n=== Algorithm Comparison ===" << std::endl;
    
    struct AlgorithmResult {
        std::string name;
        int final_size;
        double time_ms;
        double reduction;
    };
    
    std::vector<AlgorithmResult> results;
    int initial_size = obdd_count_nodes(test_bdd);
    
    // Test each algorithm and collect results
    std::vector<std::pair<ReorderStrategy, std::string>> algorithms = {
        {REORDER_SIFTING, "Sifting"},
        {REORDER_WINDOW_DP, "Window DP"},
        {REORDER_SIMULATED_ANNEALING, "Simulated Annealing"},
        {REORDER_GENETIC, "Genetic Algorithm"}
    };
    
    for (const auto& [strategy, name] : algorithms) {
        // Create fresh copy
        int order[4] = {0, 1, 2, 3};
        OBDD* bdd_copy = obdd_create(4, order);
        bdd_copy->root = test_bdd->root;
        
        ReorderConfig config = obdd_reorder_get_default_config(strategy);
        // Reduce iterations for speed
        if (strategy == REORDER_SIFTING) config.max_iterations = 3;
        if (strategy == REORDER_SIMULATED_ANNEALING) config.max_iterations = 50;
        if (strategy == REORDER_GENETIC) {
            config.population_size = 20;
            config.max_iterations = 5;
        }
        if (strategy == REORDER_WINDOW_DP) config.window_size = 2;
        
        ReorderResult result = {};
        int* new_order = obdd_reorder_advanced(bdd_copy, &config, &result);
        
        if (new_order) {
            results.push_back({
                name,
                result.final_size,
                result.execution_time_ms,
                result.reduction_ratio * 100.0
            });
            std::free(new_order);
        }
        
        obdd_destroy(bdd_copy);
    }
    
    // Print comparison table
    std::cout << "Initial BDD size: " << initial_size << " nodes" << std::endl;
    std::cout << std::string(70, '-') << std::endl;
    std::cout << "Algorithm             | Final Size | Time (ms) | Reduction (%)" << std::endl;
    std::cout << std::string(70, '-') << std::endl;
    
    for (const auto& result : results) {
        std::cout << std::setfill(' ') << std::left 
                  << std::setw(22) << result.name
                  << "| " << std::setw(11) << result.final_size
                  << "| " << std::setw(10) << std::fixed << std::setprecision(2) << result.time_ms
                  << "| " << std::setw(12) << result.reduction << std::endl;
    }
    
    std::cout << std::string(70, '-') << std::endl;
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}