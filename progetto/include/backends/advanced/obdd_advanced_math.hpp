/**
 * @file obdd_advanced_math.hpp
 * @brief Advanced Algorithms and Mathematical Applications
 * 
 * This file is part of the high-performance OBDD library providing
 * comprehensive Binary Decision Diagram operations with multi-backend
 * support for Sequential CPU, OpenMP Parallel, and CUDA GPU execution.
 * 
 * @author @vijsh32
 * @date August 13, 2024
 * @version 2.1
 * @copyright 2024 High Performance Computing Laboratory
 */


#pragma once
#ifndef OBDD_ADVANCED_MATH_HPP
#define OBDD_ADVANCED_MATH_HPP

/**
 * @file obdd_advanced_math.hpp
 * @brief Advanced mathematical applications using OBDD
 * 
 * This module implements complex mathematical problems as OBDD constructions:
 * - Modular arithmetic and number theory
 * - Cryptographic functions and S-boxes
 * - Diophantine equations
 * - Combinatorial optimization problems
 * - Boolean satisfiability problems
 * 
 * These demonstrate the power of BDD representations for complex constraints.
 */

#include "obdd.hpp"
#include <vector>
#include <utility>
#include <string>
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

/* =====================================================
   MODULAR ARITHMETIC
   ===================================================== */

/**
 * @brief Construct BDD for modular arithmetic constraint: x² + y² ≡ z² (mod p)
 * @param bits Number of bits per variable (x, y, z will each use this many bits)
 * @param modulus The modulus p
 * @return OBDD representing all valid assignments
 */
OBDD* obdd_modular_pythagorean(int bits, int modulus);

/**
 * @brief Construct BDD for modular multiplication: x * y ≡ z (mod p)
 * @param bits Number of bits per variable
 * @param modulus The modulus p
 * @return OBDD representing x * y ≡ z (mod p)
 */
OBDD* obdd_modular_multiply(int bits, int modulus);

/**
 * @brief Construct BDD for discrete logarithm: g^x ≡ y (mod p)
 * @param bits Number of bits for x and y
 * @param base The base g
 * @param modulus The prime modulus p
 * @return OBDD representing discrete log constraint
 */
OBDD* obdd_discrete_log(int bits, int base, int modulus);

/**
 * @brief Construct BDD for modular exponentiation: x^e ≡ y (mod p)
 * @param bits Number of bits for x and y
 * @param exponent The fixed exponent e
 * @param modulus The prime modulus p
 * @return OBDD representing modular exponentiation constraint
 */
OBDD* obdd_modular_exponentiation(int bits, int exponent, int modulus);

/**
 * @brief Construct BDD for quadratic residue: x² ≡ a (mod p)
 * @param bits Number of bits for x
 * @param residue The quadratic residue a
 * @param modulus The prime modulus p
 * @return OBDD representing quadratic residue constraint
 */
OBDD* obdd_quadratic_residue(int bits, int residue, int modulus);

/**
 * @brief Construct BDD for elliptic curve points: y² ≡ x³ + ax + b (mod p)
 * @param bits Number of bits for x and y coordinates
 * @param a Elliptic curve parameter a
 * @param b Elliptic curve parameter b
 * @param modulus The prime modulus p
 * @return OBDD representing valid elliptic curve points
 */
OBDD* obdd_elliptic_curve_points(int bits, int a, int b, int modulus);

/**
 * @brief Construct BDD for system of congruences (Chinese Remainder Theorem)
 * @param bits Number of bits for the solution variable
 * @param remainders Array of remainders r_i
 * @param moduli Array of moduli m_i (should be pairwise coprime)
 * @param num_congruences Number of congruences in the system
 * @return OBDD representing solutions to x ≡ r_i (mod m_i) for all i
 */
OBDD* obdd_congruence_system(int bits, int* remainders, int* moduli, int num_congruences);

/* =====================================================
   CRYPTOGRAPHIC FUNCTIONS
   ===================================================== */

/**
 * @brief Construct BDD for AES S-box transformation
 * @return OBDD with 16 variables (8 input, 8 output bits) representing AES S-box
 */
OBDD* obdd_aes_sbox(void);

/**
 * @brief Construct BDD for SHA-1 choice function: Ch(x,y,z) = (x ∧ y) ⊕ (~x ∧ z)
 * @param word_bits Number of bits per word (typically 32 for SHA-1)
 * @return OBDD representing SHA-1 choice function
 */
OBDD* obdd_sha1_choice(int word_bits);

/**
 * @brief Construct BDD for SHA-1 majority function: Maj(x,y,z) = (x ∧ y) ⊕ (x ∧ z) ⊕ (y ∧ z)
 * @param word_bits Number of bits per word
 * @return OBDD representing SHA-1 majority function
 */
OBDD* obdd_sha1_majority(int word_bits);

/**
 * @brief Construct BDD for DES S-box (simplified 4x4 version)
 * @param sbox_num S-box number (0-7)
 * @return OBDD with 10 variables (6 input, 4 output bits)
 */
OBDD* obdd_des_sbox(int sbox_num);

/**
 * @brief Construct BDD for MD5 F function: F(x,y,z) = (x ∧ y) ∨ (~x ∧ z)
 * @param word_bits Number of bits per word
 * @return OBDD representing MD5 F function
 */
OBDD* obdd_md5_f_function(int word_bits);

/**
 * @brief Construct BDD for MD5 G function: G(x,y,z) = (x ∧ z) ∨ (y ∧ ~z)
 * @param word_bits Number of bits per word
 * @return OBDD representing MD5 G function
 */
OBDD* obdd_md5_g_function(int word_bits);

/**
 * @brief Construct BDD for MD5 H function: H(x,y,z) = x ⊕ y ⊕ z
 * @param word_bits Number of bits per word
 * @return OBDD representing MD5 H function
 */
OBDD* obdd_md5_h_function(int word_bits);

/**
 * @brief Construct BDD for RSA encryption: m^e ≡ c (mod n)
 * @param bits Number of bits for message and ciphertext
 * @param exponent Public exponent e
 * @param modulus RSA modulus n
 * @return OBDD representing RSA encryption constraint
 */
OBDD* obdd_rsa_encrypt(int bits, int exponent, int modulus);

/**
 * @brief Construct BDD for Blowfish Feistel function (simplified)
 * @param bits Number of bits for input/output
 * @return OBDD representing simplified Blowfish F-function
 */
OBDD* obdd_blowfish_feistel(int bits);

/**
 * @brief Construct BDD for CRC polynomial division
 * @param bits Number of data bits
 * @param polynomial CRC polynomial (generator)
 * @return OBDD representing CRC calculation
 */
OBDD* obdd_crc_polynomial(int bits, int polynomial);

/**
 * @brief Construct BDD for ECC point addition
 * @param bits Number of bits for coordinates
 * @param a Elliptic curve parameter a
 * @param b Elliptic curve parameter b  
 * @param modulus Prime modulus p
 * @return OBDD representing point addition on elliptic curve
 */
OBDD* obdd_ecc_point_addition(int bits, int a, int b, int modulus);

/* =====================================================
   DIOPHANTINE EQUATIONS
   ===================================================== */

/**
 * @brief Construct BDD for linear Diophantine equation: ax + by = c
 * @param bits Number of bits per variable
 * @param a Coefficient of x
 * @param b Coefficient of y  
 * @param c Constant term
 * @return OBDD representing all integer solutions within bit range
 */
OBDD* obdd_linear_diophantine(int bits, int a, int b, int c);

/**
 * @brief Construct BDD for Pell equation: x² - Dy² = 1
 * @param bits Number of bits per variable
 * @param D Non-square constant
 * @return OBDD representing solutions to Pell equation
 */
OBDD* obdd_pell_equation(int bits, int D);

/**
 * @brief Construct BDD for Pythagorean triples: x² + y² = z²
 * @param bits Number of bits per variable
 * @return OBDD representing Pythagorean triples
 */
OBDD* obdd_pythagorean_triples(int bits);

/* =====================================================
   COMBINATORIAL PROBLEMS
   ===================================================== */

/**
 * @brief Construct BDD for N-Queens problem
 * @param n Board size (n x n)
 * @return OBDD with n² variables representing valid queen placements
 */
OBDD* obdd_n_queens(int n);

/**
 * @brief Construct BDD for Graph 3-coloring problem
 * @param num_vertices Number of vertices
 * @param edges Array of vertex pairs representing edges
 * @param num_edges Number of edges
 * @return OBDD representing valid 3-colorings
 */
OBDD* obdd_graph_3_coloring(int num_vertices, int (*edges)[2], int num_edges);

/**
 * @brief Construct BDD for Hamiltonian path problem
 * @param num_vertices Number of vertices
 * @param adjacency_matrix Adjacency matrix (flattened)
 * @return OBDD representing valid Hamiltonian paths
 */
OBDD* obdd_hamiltonian_path(int num_vertices, int* adjacency_matrix);

/**
 * @brief Construct BDD for Knapsack problem (0-1 version)
 * @param num_items Number of items
 * @param weights Array of item weights
 * @param values Array of item values
 * @param capacity Knapsack capacity
 * @param min_value Minimum value requirement
 * @return OBDD representing valid knapsack solutions
 */
OBDD* obdd_knapsack(int num_items, int* weights, int* values, int capacity, int min_value);

/* =====================================================
   BOOLEAN SATISFIABILITY
   ===================================================== */

/**
 * @brief Construct BDD from CNF formula
 * @param num_vars Number of variables
 * @param clauses Array of clauses (each clause is array of literals, 0-terminated)
 * @param num_clauses Number of clauses
 * @return OBDD representing satisfying assignments
 */
OBDD* obdd_from_cnf(int num_vars, int** clauses, int num_clauses);

/**
 * @brief Construct BDD for random 3-SAT instance
 * @param num_vars Number of variables
 * @param num_clauses Number of clauses
 * @param seed Random seed
 * @return OBDD representing 3-SAT instance
 */
OBDD* obdd_random_3sat(int num_vars, int num_clauses, unsigned int seed);

/**
 * @brief Construct BDD for Sudoku puzzle
 * @param puzzle 9x9 array with given clues (0 for empty cells)
 * @return OBDD representing valid Sudoku completions
 */
OBDD* obdd_sudoku(int puzzle[9][9]);

/* =====================================================
   UTILITY AND BENCHMARKING
   ===================================================== */

/**
 * @brief Count nodes in OBDD
 * @param bdd The BDD to analyze
 * @return Number of nodes in the BDD
 */
int obdd_count_nodes(const OBDD* bdd);

/**
 * @brief Count satisfying assignments in OBDD
 * @param bdd The BDD to analyze
 * @return Number of satisfying assignments (limited to 64-bit)
 */
uint64_t obdd_count_solutions(const OBDD* bdd);

/**
 * @brief Enumerate first k satisfying assignments
 * @param bdd The BDD to enumerate
 * @param assignments Output array (caller must allocate)
 * @param max_assignments Maximum number to enumerate
 * @return Actual number of assignments found
 */
int obdd_enumerate_solutions(const OBDD* bdd, int** assignments, int max_assignments);

/**
 * @brief Benchmark structure for advanced tests
 */
typedef struct {
    const char* problem_name;
    int problem_size;
    int num_variables;
    int bdd_size;
    double construction_time_ms;
    uint64_t num_solutions;
    double reordering_time_ms;
    int optimized_bdd_size;
    double reduction_ratio;
} AdvancedBenchmark;

/**
 * @brief Run comprehensive benchmark suite
 * @param results Output array for benchmark results
 * @param max_results Maximum number of results to store
 * @return Number of benchmarks completed
 */
int obdd_run_advanced_benchmarks(AdvancedBenchmark* results, int max_results);

/**
 * @brief Print benchmark results in tabular format
 */
void obdd_print_benchmark_results(const AdvancedBenchmark* results, int num_results);

#ifdef __cplusplus
}
#endif

#endif /* OBDD_ADVANCED_MATH_HPP */