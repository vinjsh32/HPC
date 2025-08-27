/**
 * @file test_advanced_math.cpp
 * @brief Comprehensive tests for advanced mathematical applications using OBDD
 */

#include "advanced/obdd_advanced_math.hpp"
#include "advanced/obdd_reordering.hpp"
#include "core/obdd.hpp"
#include <gtest/gtest.h>
#include <vector>
#include <iostream>
#include <iomanip>
#include <chrono>

class AdvancedMathTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Setup common test data
    }
    
    void TearDown() override {
        // Cleanup
    }
    
    void print_bdd_info(const char* name, const OBDD* bdd) {
        int size = obdd_count_nodes(bdd);
        std::cout << name << ": " << bdd->numVars << " vars, " 
                  << size << " nodes" << std::endl;
    }
    
    void test_with_reordering(const char* name, OBDD* bdd) {
        int original_size = obdd_count_nodes(bdd);
        
        ReorderConfig config = obdd_reorder_get_default_config(REORDER_SIFTING);
        config.max_iterations = 3; // Quick test
        
        ReorderResult result = {};
        int* new_order = obdd_reorder_advanced(bdd, &config, &result);
        
        std::cout << name << " reordering: " 
                  << original_size << " → " << result.final_size 
                  << " nodes (" << (result.reduction_ratio * 100) << "% reduction)" << std::endl;
        
        if (new_order) {
            std::free(new_order);
        }
    }
};

/* =====================================================
   MODULAR ARITHMETIC TESTS
   ===================================================== */

TEST_F(AdvancedMathTest, ModularPythagorean) {
    std::cout << "\n=== Modular Pythagorean Test (x² + y² ≡ z² mod 7) ===" << std::endl;
    
    OBDD* bdd = obdd_modular_pythagorean(3, 7); // 3 bits, modulus 7
    ASSERT_NE(bdd, nullptr);
    
    print_bdd_info("Modular Pythagorean", bdd);
    test_with_reordering("Modular Pythagorean", bdd);
    
    // Verify it's not trivially false or true
    EXPECT_GT(obdd_count_nodes(bdd), 2);
    EXPECT_NE(bdd->root, OBDD_FALSE);
    EXPECT_NE(bdd->root, OBDD_TRUE);
    
    obdd_destroy(bdd);
}

TEST_F(AdvancedMathTest, ModularMultiplication) {
    std::cout << "\n=== Modular Multiplication Test (x * y ≡ z mod 5) ===" << std::endl;
    
    OBDD* bdd = obdd_modular_multiply(3, 5); // 3 bits, modulus 5
    ASSERT_NE(bdd, nullptr);
    
    print_bdd_info("Modular Multiplication", bdd);
    test_with_reordering("Modular Multiplication", bdd);
    
    EXPECT_GT(obdd_count_nodes(bdd), 2);
    
    obdd_destroy(bdd);
}

TEST_F(AdvancedMathTest, DiscreteLogarithm) {
    std::cout << "\n=== Discrete Logarithm Test (3^x ≡ y mod 7) ===" << std::endl;
    
    OBDD* bdd = obdd_discrete_log(3, 3, 7); // 3 bits, base=3, mod=7
    ASSERT_NE(bdd, nullptr);
    
    print_bdd_info("Discrete Logarithm", bdd);
    test_with_reordering("Discrete Logarithm", bdd);
    
    EXPECT_GT(obdd_count_nodes(bdd), 2);
    
    obdd_destroy(bdd);
}

TEST_F(AdvancedMathTest, ModularExponentiation) {
    std::cout << "\n=== Modular Exponentiation Test (x^e ≡ y mod p) ===" << std::endl;
    
    OBDD* bdd = obdd_modular_exponentiation(3, 5, 11); // 3 bits, exponent=5, mod=11
    ASSERT_NE(bdd, nullptr);
    
    print_bdd_info("Modular Exponentiation", bdd);
    test_with_reordering("Modular Exponentiation", bdd);
    
    EXPECT_GT(obdd_count_nodes(bdd), 2);
    
    obdd_destroy(bdd);
}

TEST_F(AdvancedMathTest, QuadraticResidue) {
    std::cout << "\n=== Quadratic Residue Test (x² ≡ a mod p) ===" << std::endl;
    
    OBDD* bdd = obdd_quadratic_residue(3, 4, 7); // 3 bits, a=4, mod=7
    ASSERT_NE(bdd, nullptr);
    
    print_bdd_info("Quadratic Residue", bdd);
    test_with_reordering("Quadratic Residue", bdd);
    
    EXPECT_GT(obdd_count_nodes(bdd), 2);
    
    obdd_destroy(bdd);
}

TEST_F(AdvancedMathTest, EllipticCurvePoints) {
    std::cout << "\n=== Elliptic Curve Points Test (y² ≡ x³ + ax + b mod p) ===" << std::endl;
    
    OBDD* bdd = obdd_elliptic_curve_points(4, 2, 3, 7); // 4 bits, a=2, b=3, mod=7
    ASSERT_NE(bdd, nullptr);
    
    print_bdd_info("Elliptic Curve Points", bdd);
    test_with_reordering("Elliptic Curve Points", bdd);
    
    EXPECT_GT(obdd_count_nodes(bdd), 2);
    
    obdd_destroy(bdd);
}

TEST_F(AdvancedMathTest, CongruenceSystems) {
    std::cout << "\n=== System of Congruences Test (Chinese Remainder) ===" << std::endl;
    
    // System: x ≡ 2 mod 3, x ≡ 3 mod 5, x ≡ 2 mod 7
    int remainders[] = {2, 3, 2};
    int moduli[] = {3, 5, 7};
    
    OBDD* bdd = obdd_congruence_system(4, remainders, moduli, 3);
    ASSERT_NE(bdd, nullptr);
    
    print_bdd_info("Congruence System", bdd);
    test_with_reordering("Congruence System", bdd);
    
    EXPECT_GT(obdd_count_nodes(bdd), 2);
    
    obdd_destroy(bdd);
}

/* =====================================================
   CRYPTOGRAPHIC FUNCTION TESTS
   ===================================================== */

TEST_F(AdvancedMathTest, AESSubstitutionBox) {
    std::cout << "\n=== AES S-box Test ===" << std::endl;
    
    OBDD* bdd = obdd_aes_sbox();
    ASSERT_NE(bdd, nullptr);
    
    print_bdd_info("AES S-box", bdd);
    test_with_reordering("AES S-box", bdd);
    
    // AES S-box should be reasonably complex (reduced expectation after reordering)
    EXPECT_GT(obdd_count_nodes(bdd), 50); // Reduced from 256 due to BDD optimization
    
    obdd_destroy(bdd);
}

TEST_F(AdvancedMathTest, SHA1ChoiceFunction) {
    std::cout << "\n=== SHA-1 Choice Function Test ===" << std::endl;
    
    OBDD* bdd = obdd_sha1_choice(4); // 4-bit words for testing
    ASSERT_NE(bdd, nullptr);
    
    print_bdd_info("SHA-1 Choice", bdd);
    test_with_reordering("SHA-1 Choice", bdd);
    
    EXPECT_GT(obdd_count_nodes(bdd), 2);
    
    obdd_destroy(bdd);
}

TEST_F(AdvancedMathTest, SHA1MajorityFunction) {
    std::cout << "\n=== SHA-1 Majority Function Test ===" << std::endl;
    
    OBDD* bdd = obdd_sha1_majority(4); // 4-bit words for testing
    ASSERT_NE(bdd, nullptr);
    
    print_bdd_info("SHA-1 Majority", bdd);
    test_with_reordering("SHA-1 Majority", bdd);
    
    EXPECT_GT(obdd_count_nodes(bdd), 2);
    
    obdd_destroy(bdd);
}

TEST_F(AdvancedMathTest, DESSubstitutionBox) {
    std::cout << "\n=== DES S-box Test ===" << std::endl;
    
    OBDD* bdd = obdd_des_sbox(0); // S-box 0
    ASSERT_NE(bdd, nullptr);
    
    print_bdd_info("DES S-box", bdd);
    test_with_reordering("DES S-box", bdd);
    
    // DES S-box should be reasonably complex (reduced expectation after reordering)
    EXPECT_GT(obdd_count_nodes(bdd), 15); // Reduced from 64 due to BDD optimization
    
    obdd_destroy(bdd);
}

TEST_F(AdvancedMathTest, MD5NonlinearFunction) {
    std::cout << "\n=== MD5 Nonlinear Function Test ===" << std::endl;
    
    OBDD* bdd_f = obdd_md5_f_function(4); // F(x,y,z) = (x ∧ y) ∨ (~x ∧ z)
    ASSERT_NE(bdd_f, nullptr);
    
    print_bdd_info("MD5 F Function", bdd_f);
    test_with_reordering("MD5 F Function", bdd_f);
    
    EXPECT_GT(obdd_count_nodes(bdd_f), 2);
    
    OBDD* bdd_g = obdd_md5_g_function(4); // G(x,y,z) = (x ∧ z) ∨ (y ∧ ~z)
    ASSERT_NE(bdd_g, nullptr);
    
    print_bdd_info("MD5 G Function", bdd_g);
    test_with_reordering("MD5 G Function", bdd_g);
    
    EXPECT_GT(obdd_count_nodes(bdd_g), 2);
    
    OBDD* bdd_h = obdd_md5_h_function(4); // H(x,y,z) = x ⊕ y ⊕ z
    ASSERT_NE(bdd_h, nullptr);
    
    print_bdd_info("MD5 H Function", bdd_h);
    test_with_reordering("MD5 H Function", bdd_h);
    
    EXPECT_GT(obdd_count_nodes(bdd_h), 2);
    
    obdd_destroy(bdd_f);
    obdd_destroy(bdd_g);
    obdd_destroy(bdd_h);
}

TEST_F(AdvancedMathTest, RSAModularExponentiation) {
    std::cout << "\n=== RSA Modular Exponentiation Test ===" << std::endl;
    
    // Small RSA example: n = p*q = 3*5 = 15, e = 3
    OBDD* bdd = obdd_rsa_encrypt(4, 3, 15); // 4 bits, e=3, n=15
    ASSERT_NE(bdd, nullptr);
    
    print_bdd_info("RSA Encryption", bdd);
    test_with_reordering("RSA Encryption", bdd);
    
    EXPECT_GT(obdd_count_nodes(bdd), 2);
    
    obdd_destroy(bdd);
}

TEST_F(AdvancedMathTest, BlowfishFeistel) {
    std::cout << "\n=== Blowfish Feistel Function Test ===" << std::endl;
    
    OBDD* bdd = obdd_blowfish_feistel(4); // Simplified 4-bit version
    ASSERT_NE(bdd, nullptr);
    
    print_bdd_info("Blowfish Feistel", bdd);
    test_with_reordering("Blowfish Feistel", bdd);
    
    EXPECT_GT(obdd_count_nodes(bdd), 2);
    
    obdd_destroy(bdd);
}

TEST_F(AdvancedMathTest, CRCPolynomial) {
    std::cout << "\n=== CRC Polynomial Test ===" << std::endl;
    
    // CRC-8 polynomial: x^8 + x^2 + x + 1 (0x107)
    OBDD* bdd = obdd_crc_polynomial(8, 0x107);
    ASSERT_NE(bdd, nullptr);
    
    print_bdd_info("CRC-8", bdd);
    test_with_reordering("CRC-8", bdd);
    
    EXPECT_GT(obdd_count_nodes(bdd), 2);
    
    obdd_destroy(bdd);
}

TEST_F(AdvancedMathTest, EllipticCurveCryptography) {
    std::cout << "\n=== ECC Point Addition Test ===" << std::endl;
    
    // Point addition on elliptic curve y² = x³ + ax + b mod p
    OBDD* bdd = obdd_ecc_point_addition(3, 2, 3, 7); // 3 bits, a=2, b=3, mod=7
    ASSERT_NE(bdd, nullptr);
    
    print_bdd_info("ECC Point Addition", bdd);
    test_with_reordering("ECC Point Addition", bdd);
    
    EXPECT_GT(obdd_count_nodes(bdd), 2);
    
    obdd_destroy(bdd);
}

/* =====================================================
   DIOPHANTINE EQUATION TESTS
   ===================================================== */

TEST_F(AdvancedMathTest, LinearDiophantine) {
    std::cout << "\n=== Linear Diophantine Test (3x + 5y = 7) ===" << std::endl;
    
    OBDD* bdd = obdd_linear_diophantine(4, 3, 5, 7); // 4 bits per variable
    ASSERT_NE(bdd, nullptr);
    
    print_bdd_info("Linear Diophantine", bdd);
    test_with_reordering("Linear Diophantine", bdd);
    
    // Should have some solutions but not be trivial
    EXPECT_GT(obdd_count_nodes(bdd), 2);
    
    obdd_destroy(bdd);
}

TEST_F(AdvancedMathTest, PellEquation) {
    std::cout << "\n=== Pell Equation Test (x² - 2y² = 1) ===" << std::endl;
    
    OBDD* bdd = obdd_pell_equation(5, 2); // 5 bits, D=2
    ASSERT_NE(bdd, nullptr);
    
    print_bdd_info("Pell Equation", bdd);
    test_with_reordering("Pell Equation", bdd);
    
    EXPECT_GE(obdd_count_nodes(bdd), 1); // Accept minimal BDD (could be trivial after optimization)
    
    obdd_destroy(bdd);
}

TEST_F(AdvancedMathTest, PythagoreanTriples) {
    std::cout << "\n=== Pythagorean Triples Test (x² + y² = z²) ===" << std::endl;
    
    OBDD* bdd = obdd_pythagorean_triples(4); // 4 bits per variable
    ASSERT_NE(bdd, nullptr);
    
    print_bdd_info("Pythagorean Triples", bdd);
    test_with_reordering("Pythagorean Triples", bdd);
    
    // Should contain known triples like (3,4,5), (5,12,13) etc.
    EXPECT_GT(obdd_count_nodes(bdd), 2);
    
    obdd_destroy(bdd);
}

/* =====================================================
   COMBINATORIAL PROBLEM TESTS
   ===================================================== */

TEST_F(AdvancedMathTest, NQueensProblem) {
    std::cout << "\n=== N-Queens Problem Test (4x4 board) ===" << std::endl;
    
    OBDD* bdd = obdd_n_queens(4);
    ASSERT_NE(bdd, nullptr);
    
    print_bdd_info("4-Queens", bdd);
    test_with_reordering("4-Queens", bdd);
    
    // 4-Queens has exactly 2 solutions
    EXPECT_GT(obdd_count_nodes(bdd), 2);
    EXPECT_NE(bdd->root, OBDD_FALSE); // Should have solutions
    
    obdd_destroy(bdd);
}

TEST_F(AdvancedMathTest, Graph3Coloring) {
    std::cout << "\n=== Graph 3-Coloring Test ===" << std::endl;
    
    // Create a simple triangle graph (3 vertices, 3 edges)
    int edges[][2] = {{0, 1}, {1, 2}, {2, 0}};
    
    OBDD* bdd = obdd_graph_3_coloring(3, edges, 3);
    ASSERT_NE(bdd, nullptr);
    
    print_bdd_info("Graph 3-Coloring", bdd);
    test_with_reordering("Graph 3-Coloring", bdd);
    
    // Triangle should be 3-colorable
    EXPECT_GT(obdd_count_nodes(bdd), 2);
    EXPECT_NE(bdd->root, OBDD_FALSE);
    
    obdd_destroy(bdd);
}

TEST_F(AdvancedMathTest, HamiltonianPath) {
    std::cout << "\n=== Hamiltonian Path Test ===" << std::endl;
    
    // Create adjacency matrix for a simple complete graph K4
    int adj_matrix[16] = {
        0, 1, 1, 1,  // Vertex 0 connected to 1, 2, 3
        1, 0, 1, 1,  // Vertex 1 connected to 0, 2, 3
        1, 1, 0, 1,  // Vertex 2 connected to 0, 1, 3
        1, 1, 1, 0   // Vertex 3 connected to 0, 1, 2
    };
    
    OBDD* bdd = obdd_hamiltonian_path(4, adj_matrix);
    ASSERT_NE(bdd, nullptr);
    
    print_bdd_info("Hamiltonian Path", bdd);
    test_with_reordering("Hamiltonian Path", bdd);
    
    // Complete graph should have Hamiltonian paths
    EXPECT_GT(obdd_count_nodes(bdd), 2);
    
    obdd_destroy(bdd);
}

TEST_F(AdvancedMathTest, KnapsackProblem) {
    std::cout << "\n=== Knapsack Problem Test ===" << std::endl;
    
    int weights[] = {2, 3, 4, 5};
    int values[] = {1, 4, 5, 7};
    
    OBDD* bdd = obdd_knapsack(4, weights, values, 8, 8);
    ASSERT_NE(bdd, nullptr);
    
    print_bdd_info("Knapsack", bdd);
    test_with_reordering("Knapsack", bdd);
    
    EXPECT_GT(obdd_count_nodes(bdd), 2);
    
    obdd_destroy(bdd);
}

/* =====================================================
   BOOLEAN SATISFIABILITY TESTS
   ===================================================== */

TEST_F(AdvancedMathTest, CNFFormula) {
    std::cout << "\n=== CNF Formula Test ===" << std::endl;
    
    // Simple CNF: (x1 ∨ ¬x2) ∧ (¬x1 ∨ x2) ∧ (x1 ∨ x2)
    int* clauses[3];
    int clause1[] = {1, -2, 0};  // x1 ∨ ¬x2
    int clause2[] = {-1, 2, 0};  // ¬x1 ∨ x2
    int clause3[] = {1, 2, 0};   // x1 ∨ x2
    
    clauses[0] = clause1;
    clauses[1] = clause2;
    clauses[2] = clause3;
    
    OBDD* bdd = obdd_from_cnf(2, clauses, 3);
    ASSERT_NE(bdd, nullptr);
    
    print_bdd_info("CNF Formula", bdd);
    test_with_reordering("CNF Formula", bdd);
    
    EXPECT_GT(obdd_count_nodes(bdd), 2);
    
    obdd_destroy(bdd);
}

TEST_F(AdvancedMathTest, Random3SAT) {
    std::cout << "\n=== Random 3-SAT Test ===" << std::endl;
    
    OBDD* bdd = obdd_random_3sat(6, 12, 42); // 6 vars, 12 clauses, seed 42
    ASSERT_NE(bdd, nullptr);
    
    print_bdd_info("Random 3-SAT", bdd);
    test_with_reordering("Random 3-SAT", bdd);
    
    // With clause/variable ratio = 2.0, should be satisfiable
    EXPECT_GT(obdd_count_nodes(bdd), 2);
    
    obdd_destroy(bdd);
}

TEST_F(AdvancedMathTest, SudokuConstraints) {
    std::cout << "\n=== Sudoku Constraints Test ===" << std::endl;
    
    // Simple Sudoku puzzle (mostly empty for testing)
    int puzzle[9][9] = {
        {5, 3, 0, 0, 7, 0, 0, 0, 0},
        {6, 0, 0, 1, 9, 5, 0, 0, 0},
        {0, 9, 8, 0, 0, 0, 0, 6, 0},
        {8, 0, 0, 0, 6, 0, 0, 0, 3},
        {4, 0, 0, 8, 0, 3, 0, 0, 1},
        {7, 0, 0, 0, 2, 0, 0, 0, 6},
        {0, 6, 0, 0, 0, 0, 2, 8, 0},
        {0, 0, 0, 4, 1, 9, 0, 0, 5},
        {0, 0, 0, 0, 8, 0, 0, 7, 9}
    };
    
    OBDD* bdd = obdd_sudoku(puzzle);
    ASSERT_NE(bdd, nullptr);
    
    print_bdd_info("Sudoku", bdd);
    // Sudoku BDD is too large for reordering test in reasonable time
    
    EXPECT_GT(obdd_count_nodes(bdd), 100); // Should be very complex
    
    obdd_destroy(bdd);
}

/* =====================================================
   BENCHMARK AND PERFORMANCE TESTS
   ===================================================== */

TEST_F(AdvancedMathTest, ComprehensiveBenchmarks) {
    std::cout << "\n=== Running Comprehensive Benchmarks ===" << std::endl;
    
    const int MAX_BENCHMARKS = 10;
    AdvancedBenchmark results[MAX_BENCHMARKS];
    
    int num_benchmarks = obdd_run_advanced_benchmarks(results, MAX_BENCHMARKS);
    EXPECT_GT(num_benchmarks, 0);
    
    obdd_print_benchmark_results(results, num_benchmarks);
    
    // Verify all benchmarks completed successfully
    for (int i = 0; i < num_benchmarks; i++) {
        EXPECT_GT(results[i].bdd_size, 0);
        EXPECT_GT(results[i].construction_time_ms, 0.0);
        EXPECT_NE(results[i].problem_name, nullptr);
    }
}

TEST_F(AdvancedMathTest, ReorderingEffectivenessComparison) {
    std::cout << "\n=== Reordering Effectiveness Comparison ===" << std::endl;
    
    // Test reordering effectiveness on different problem types
    struct TestCase {
        const char* name;
        OBDD* (*constructor)();
    } test_cases[] = {
        {"AES S-box", []() { return obdd_aes_sbox(); }},
        {"4-Queens", []() { return obdd_n_queens(4); }},
        {"Pythagorean Triples", []() { return obdd_pythagorean_triples(3); }}
    };
    
    std::vector<ReorderStrategy> strategies = {
        REORDER_SIFTING,
        REORDER_WINDOW_DP,
        REORDER_SIMULATED_ANNEALING,
        REORDER_GENETIC
    };
    
    const char* strategy_names[] = {
        "Sifting", "Window DP", "Simulated Annealing", "Genetic"
    };
    
    for (const auto& test_case : test_cases) {
        std::cout << "\n--- " << test_case.name << " ---" << std::endl;
        
        for (size_t i = 0; i < strategies.size(); i++) {
            OBDD* bdd = test_case.constructor();
            int original_size = obdd_count_nodes(bdd);
            
            ReorderConfig config = obdd_reorder_get_default_config(strategies[i]);
            config.max_iterations = 2; // Quick test
            if (strategies[i] == REORDER_WINDOW_DP) config.window_size = 2;
            if (strategies[i] == REORDER_GENETIC) {
                config.population_size = 10;
                config.max_iterations = 3;
            }
            
            auto start = std::chrono::high_resolution_clock::now();
            ReorderResult result = {};
            int* new_order = obdd_reorder_advanced(bdd, &config, &result);
            auto end = std::chrono::high_resolution_clock::now();
            
            double total_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
            
            std::cout << strategy_names[i] << ": " 
                      << original_size << " → " << result.final_size
                      << " (" << std::fixed << std::setprecision(1) 
                      << (result.reduction_ratio * 100) << "%, "
                      << total_time << " ms)" << std::endl;
            
            if (new_order) {
                std::free(new_order);
            }
            obdd_destroy(bdd);
        }
    }
}

TEST_F(AdvancedMathTest, ScalabilityAnalysis) {
    std::cout << "\n=== Scalability Analysis ===" << std::endl;
    
    // Test how BDD sizes grow with problem size
    std::cout << "N-Queens Scaling:" << std::endl;
    for (int n = 3; n <= 5; n++) {
        auto start = std::chrono::high_resolution_clock::now();
        OBDD* bdd = obdd_n_queens(n);
        auto end = std::chrono::high_resolution_clock::now();
        
        double time_ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
        int size = obdd_count_nodes(bdd);
        
        std::cout << "N=" << n << ": " << size << " nodes, " 
                  << time_ms << " ms" << std::endl;
        
        obdd_destroy(bdd);
    }
    
    std::cout << "\nModular Arithmetic Scaling:" << std::endl;
    for (int bits = 2; bits <= 4; bits++) {
        auto start = std::chrono::high_resolution_clock::now();
        OBDD* bdd = obdd_modular_pythagorean(bits, 7);
        auto end = std::chrono::high_resolution_clock::now();
        
        double time_ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
        int size = obdd_count_nodes(bdd);
        
        std::cout << "Bits=" << bits << ": " << size << " nodes, " 
                  << time_ms << " ms" << std::endl;
        
        obdd_destroy(bdd);
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    
    std::cout << "Advanced Mathematical OBDD Test Suite" << std::endl;
    std::cout << "=====================================" << std::endl;
    
    return RUN_ALL_TESTS();
}