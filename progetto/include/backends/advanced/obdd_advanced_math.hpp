/*
 * This file is part of the High-Performance OBDD Library
 * Copyright (C) 2024 High Performance Computing Laboratory
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <https://www.gnu.org/licenses/>.
 * 
 * Authors: Vincenzo Ferraro
 * Student ID: 0622702113
 * Email: v.ferraro5@studenti.unisa.it
 * Assignment: Final Project - Parallel OBDD Implementation
 * Course: High Performance Computing - Prof. Moscato
 * University: Università degli studi di Salerno - Ingegneria Informatica magistrale
 */

/**
 * @file obdd_advanced_math.hpp
 * @brief Advanced Mathematical Applications and Constraint Modeling using OBDDs
 * 
 * Corso di High Performance Computing - Prof. Moscato - Università degli studi di Salerno - Ingegneria Informatica magistrale
 * 
 * ADVANCED MATHEMATICAL CONSTRAINT MODELING:
 * ==========================================
 * This header defines an extensive library of advanced mathematical applications
 * implemented using Binary Decision Diagrams. The module demonstrates the power
 * of OBDD representations for complex mathematical constraints, cryptographic
 * functions, and combinatorial optimization problems.
 * 
 * CORE MATHEMATICAL DOMAINS:
 * ==========================
 * 
 * 1. MODULAR ARITHMETIC AND NUMBER THEORY:
 *    - Quadratic congruences and Pythagorean constraints
 *    - Discrete logarithm problems for cryptographic analysis
 *    - Elliptic curve point validation and operations
 *    - Chinese Remainder Theorem system solving
 *    - Quadratic residue computations for primality testing
 * 
 * 2. CRYPTOGRAPHIC FUNCTION MODELING:
 *    - Advanced Encryption Standard (AES) S-box transformations
 *    - SHA-1 choice and majority functions for hash analysis
 *    - Data Encryption Standard (DES) S-box implementations
 *    - MD5 function family (F, G, H) for digest computation
 *    - RSA encryption constraint modeling for security analysis
 *    - Elliptic Curve Cryptography (ECC) point arithmetic
 * 
 * 3. DIOPHANTINE EQUATION SOLVING:
 *    - Linear Diophantine equations with integer solutions
 *    - Pell equations for continued fraction analysis
 *    - Pythagorean triple generation and validation
 *    - Polynomial constraint satisfaction over integer domains
 * 
 * 4. COMBINATORIAL OPTIMIZATION PROBLEMS:
 *    - N-Queens problem for constraint satisfaction demonstration
 *    - Graph coloring problems for scheduling and resource allocation
 *    - Hamiltonian path detection for routing optimization
 *    - 0-1 Knapsack problem for resource optimization
 * 
 * 5. BOOLEAN SATISFIABILITY APPLICATIONS:
 *    - CNF formula conversion and analysis
 *    - Random 3-SAT instance generation for benchmarking
 *    - Sudoku puzzle solving through constraint encoding
 *    - SAT-based verification and model checking applications
 * 
 * MATHEMATICAL CONSTRAINT BREAKTHROUGH:
 * =====================================
 * 
 * 1. CONSTRAINT ENCODING STRATEGIES:
 *    - Direct mathematical constraint translation to Boolean logic
 *    - Efficient bit-vector arithmetic encoding for numerical problems
 *    - Modular arithmetic optimization using BDD-native operations
 *    - Polynomial constraint decomposition into BDD-friendly subproblems
 * 
 * 2. SCALABILITY OPTIMIZATION:
 *    - Variable ordering optimization for mathematical constraint BDDs
 *    - Hierarchical constraint decomposition for large problem instances
 *    - Memory-efficient constraint representation using shared sub-BDDs
 *    - Progressive constraint building for interactive problem solving
 * 
 * 3. NUMERICAL PRECISION MANAGEMENT:
 *    - Configurable precision through variable bit allocation
 *    - Overflow detection and handling in modular arithmetic
 *    - Precision-performance trade-off analysis for different domains
 *    - Error bound calculation for approximate constraint satisfaction
 * 
 * CRYPTOGRAPHIC ANALYSIS APPLICATIONS:
 * ====================================
 * 
 * 1. CRYPTOGRAPHIC PRIMITIVE ANALYSIS:
 *    - S-box linearity and differential analysis using BDD operations
 *    - Hash function collision detection through constraint modeling
 *    - Block cipher round function analysis and optimization
 *    - Stream cipher feedback function modeling and analysis
 * 
 * 2. SECURITY PARAMETER OPTIMIZATION:
 *    - Key schedule analysis for symmetric encryption algorithms
 *    - Public key cryptography parameter validation
 *    - Random number generator entropy analysis
 *    - Side-channel attack vector identification through constraint analysis
 * 
 * 3. PROTOCOL VERIFICATION:
 *    - Authentication protocol constraint modeling
 *    - Key exchange security property verification
 *    - Digital signature scheme validation through mathematical constraints
 *    - Zero-knowledge proof system constraint encoding
 * 
 * COMBINATORIAL PROBLEM SOLVING:
 * ==============================
 * 
 * 1. CONSTRAINT SATISFACTION PROBLEMS:
 *    - Systematic constraint propagation using BDD operations
 *    - Solution space exploration through BDD traversal
 *    - Optimization objective encoding as BDD constraints
 *    - Multi-objective optimization through constraint composition
 * 
 * 2. GRAPH ALGORITHMIC PROBLEMS:
 *    - Graph property verification using Boolean constraint encoding
 *    - Network flow optimization through constraint modeling
 *    - Matching problem solution enumeration using BDD operations
 *    - Shortest path constraint modeling for routing optimization
 * 
 * 3. SCHEDULING AND RESOURCE ALLOCATION:
 *    - Resource conflict detection through constraint intersection
 *    - Optimal scheduling through constraint optimization
 *    - Load balancing constraint modeling and solution
 *    - Deadline constraint satisfaction with resource optimization
 * 
 * PERFORMANCE AND BENCHMARKING:
 * =============================
 * 
 * 1. MATHEMATICAL PROBLEM COMPLEXITY:
 *    - BDD size analysis for different mathematical constraint types
 *    - Variable ordering impact on mathematical constraint BDDs
 *    - Construction time analysis for complex mathematical problems
 *    - Memory usage optimization for large-scale mathematical constraints
 * 
 * 2. SOLUTION ENUMERATION EFFICIENCY:
 *    - Satisfying assignment counting for constraint satisfaction problems
 *    - Solution enumeration algorithms optimized for BDD representation
 *    - Partial solution exploration for interactive constraint solving
 *    - Solution quality metrics and optimization objective evaluation
 * 
 * 3. BENCHMARKING METHODOLOGY:
 *    - Comprehensive benchmark suite for mathematical constraint problems
 *    - Performance comparison across different mathematical domains
 *    - Scalability analysis for increasing problem complexity
 *    - Memory and time complexity characterization for constraint types
 * 
 * INTEGRATION WITH COMPUTATIONAL BACKENDS:
 * ========================================
 * - Seamless integration with sequential, OpenMP, and CUDA backends
 * - Mathematical constraint optimization specific to each computational platform
 * - Parallel constraint evaluation for large mathematical problems
 * - GPU acceleration of mathematical constraint satisfaction algorithms
 * 
 * @author vinjsh32
 * @date September 2, 2024
 * @version 3.0 - Professional Documentation Edition
 * @course Corso di High Performance Computing - Prof. Moscato
 * @university Università degli studi di Salerno - Ingegneria Informatica magistrale
 */

#pragma once
#ifndef OBDD_ADVANCED_MATH_HPP
#define OBDD_ADVANCED_MATH_HPP

#include "obdd.hpp"
#include <vector>
#include <utility>
#include <string>
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

/* --------------------------------------------------------------------------
 * MODULAR ARITHMETIC AND NUMBER THEORY
 * 
 * Advanced number theoretic functions implemented as BDD constraint systems.
 * These functions demonstrate the power of BDD representation for complex
 * mathematical constraints involving modular arithmetic and algebraic structures.
 * -------------------------------------------------------------------------- */

/**
 * @brief Construct BDD for modular Pythagorean constraint: x² + y² ≡ z² (mod p)
 * 
 * @param bits Number of bits per variable (x, y, z will each use this many bits)
 * @param modulus The prime modulus p for the congruence relation
 * @return OBDD representing all valid assignments satisfying the constraint
 * 
 * MATHEMATICAL CONSTRAINT ENCODING:
 * This function creates a BDD that encodes the modular Pythagorean constraint,
 * which generalizes the classical Pythagorean theorem to modular arithmetic.
 * The constraint is fundamental in number theory and has applications in
 * cryptography and algebraic geometry.
 * 
 * ALGORITHMIC APPROACH:
 * - Bit-vector encoding of integer variables with specified precision
 * - Modular arithmetic implemented through BDD-native Boolean operations
 * - Constraint propagation optimized for BDD representation efficiency
 * - Variable ordering optimization for minimal BDD size
 */
OBDD* obdd_modular_pythagorean(int bits, int modulus);

/**
 * @brief Construct BDD for modular multiplication constraint: x * y ≡ z (mod p)
 * 
 * @param bits Number of bits per variable for precision control
 * @param modulus The prime modulus p for the multiplication constraint
 * @return OBDD representing the modular multiplication relation
 * 
 * MODULAR MULTIPLICATION IMPLEMENTATION:
 * Implements the fundamental modular multiplication constraint using efficient
 * BDD encoding. This constraint is central to many cryptographic algorithms
 * and number theoretic computations.
 * 
 * OPTIMIZATION STRATEGIES:
 * - Efficient modular reduction using BDD Boolean operations
 * - Overflow handling through proper bit allocation
 * - Performance optimization for common cryptographic moduli
 * - Integration with Chinese Remainder Theorem for large moduli
 */
OBDD* obdd_modular_multiply(int bits, int modulus);

/**
 * @brief Construct BDD for discrete logarithm problem: g^x ≡ y (mod p)
 * 
 * @param bits Number of bits for variables x and y
 * @param base The base g for the exponential operation
 * @param modulus The prime modulus p for the discrete logarithm
 * @return OBDD representing the discrete logarithm constraint
 * 
 * DISCRETE LOGARITHM CONSTRAINT:
 * The discrete logarithm problem is fundamental to modern public-key
 * cryptography. This function encodes the constraint as a BDD, enabling
 * analysis of small instances and cryptographic parameter validation.
 * 
 * CRYPTOGRAPHIC APPLICATIONS:
 * - Elliptic Curve Cryptography (ECC) parameter analysis
 * - Diffie-Hellman key exchange security evaluation
 * - Digital signature algorithm parameter validation
 * - Cryptographic protocol verification through constraint solving
 */
OBDD* obdd_discrete_log(int bits, int base, int modulus);

/**
 * @brief Construct BDD for modular exponentiation: x^e ≡ y (mod p)
 * 
 * @param bits Number of bits for variables x and y
 * @param exponent The fixed exponent e
 * @param modulus The prime modulus p
 * @return OBDD representing modular exponentiation constraint
 * 
 * MODULAR EXPONENTIATION ENCODING:
 * Implements efficient BDD encoding of modular exponentiation constraints.
 * Essential for RSA cryptosystem analysis and other exponential-based
 * cryptographic primitive evaluation.
 */
OBDD* obdd_modular_exponentiation(int bits, int exponent, int modulus);

/**
 * @brief Construct BDD for quadratic residue constraint: x² ≡ a (mod p)
 * 
 * @param bits Number of bits for variable x
 * @param residue The quadratic residue a
 * @param modulus The prime modulus p
 * @return OBDD representing quadratic residue constraint
 * 
 * QUADRATIC RESIDUE THEORY:
 * Quadratic residues are fundamental in number theory and cryptography.
 * This function enables analysis of quadratic residuosity problems and
 * related cryptographic constructions.
 */
OBDD* obdd_quadratic_residue(int bits, int residue, int modulus);

/**
 * @brief Construct BDD for elliptic curve point validation: y² ≡ x³ + ax + b (mod p)
 * 
 * @param bits Number of bits for x and y coordinates
 * @param a Elliptic curve parameter a
 * @param b Elliptic curve parameter b
 * @param modulus The prime modulus p
 * @return OBDD representing valid elliptic curve points
 * 
 * ELLIPTIC CURVE CRYPTOGRAPHY:
 * Implements the fundamental elliptic curve equation as a BDD constraint.
 * Essential for ECC parameter validation and cryptographic analysis.
 */
OBDD* obdd_elliptic_curve_points(int bits, int a, int b, int modulus);

/**
 * @brief Construct BDD for Chinese Remainder Theorem system
 * 
 * @param bits Number of bits for the solution variable
 * @param remainders Array of remainders r_i
 * @param moduli Array of moduli m_i (should be pairwise coprime)
 * @param num_congruences Number of congruences in the system
 * @return OBDD representing solutions to x ≡ r_i (mod m_i) for all i
 * 
 * CHINESE REMAINDER THEOREM APPLICATION:
 * Implements the classical Chinese Remainder Theorem as a BDD constraint system.
 * Enables efficient solution of systems of linear congruences.
 */
OBDD* obdd_congruence_system(int bits, int* remainders, int* moduli, int num_congruences);

/* --------------------------------------------------------------------------
 * CRYPTOGRAPHIC FUNCTIONS AND SECURITY ANALYSIS
 * 
 * Implementation of standard cryptographic functions as BDD constraints.
 * These functions enable cryptographic analysis, security evaluation, and
 * verification of cryptographic primitive properties.
 * -------------------------------------------------------------------------- */

/**
 * @brief Construct BDD for AES S-box transformation
 * 
 * @return OBDD with 16 variables (8 input, 8 output bits) representing AES S-box
 * 
 * AES S-BOX IMPLEMENTATION:
 * The Advanced Encryption Standard S-box is a crucial component for security
 * analysis. This function creates a complete BDD representation enabling
 * differential and linear cryptanalysis.
 */
OBDD* obdd_aes_sbox(void);

/**
 * @brief Construct BDD for SHA-1 choice function: Ch(x,y,z) = (x ∧ y) ⊕ (~x ∧ z)
 * 
 * @param word_bits Number of bits per word (typically 32 for SHA-1)
 * @return OBDD representing SHA-1 choice function
 * 
 * SHA-1 CRYPTOGRAPHIC ANALYSIS:
 * Implements the SHA-1 choice function for hash function analysis and
 * collision detection research.
 */
OBDD* obdd_sha1_choice(int word_bits);

/**
 * @brief Construct BDD for SHA-1 majority function: Maj(x,y,z) = (x ∧ y) ⊕ (x ∧ z) ⊕ (y ∧ z)
 * 
 * @param word_bits Number of bits per word
 * @return OBDD representing SHA-1 majority function
 * 
 * SHA-1 MAJORITY FUNCTION:
 * Essential component of SHA-1 hash function for cryptographic analysis
 * and security evaluation.
 */
OBDD* obdd_sha1_majority(int word_bits);

/**
 * @brief Construct BDD for DES S-box (6-to-4 bit transformation)
 * 
 * @param sbox_num S-box number (0-7 for the eight DES S-boxes)
 * @return OBDD with 10 variables (6 input, 4 output bits)
 * 
 * DES S-BOX ANALYSIS:
 * Data Encryption Standard S-box implementation for differential
 * cryptanalysis and security evaluation.
 */
OBDD* obdd_des_sbox(int sbox_num);

/**
 * @brief Construct BDD for MD5 F function: F(x,y,z) = (x ∧ y) ∨ (~x ∧ z)
 * 
 * @param word_bits Number of bits per word
 * @return OBDD representing MD5 F function
 */
OBDD* obdd_md5_f_function(int word_bits);

/**
 * @brief Construct BDD for MD5 G function: G(x,y,z) = (x ∧ z) ∨ (y ∧ ~z)
 * 
 * @param word_bits Number of bits per word
 * @return OBDD representing MD5 G function
 */
OBDD* obdd_md5_g_function(int word_bits);

/**
 * @brief Construct BDD for MD5 H function: H(x,y,z) = x ⊕ y ⊕ z
 * 
 * @param word_bits Number of bits per word
 * @return OBDD representing MD5 H function
 */
OBDD* obdd_md5_h_function(int word_bits);

/**
 * @brief Construct BDD for RSA encryption constraint: m^e ≡ c (mod n)
 * 
 * @param bits Number of bits for message and ciphertext
 * @param exponent Public exponent e
 * @param modulus RSA modulus n
 * @return OBDD representing RSA encryption constraint
 */
OBDD* obdd_rsa_encrypt(int bits, int exponent, int modulus);

/**
 * @brief Construct BDD for Blowfish Feistel function (simplified)
 * 
 * @param bits Number of bits for input/output
 * @return OBDD representing simplified Blowfish F-function
 */
OBDD* obdd_blowfish_feistel(int bits);

/**
 * @brief Construct BDD for CRC polynomial division
 * 
 * @param bits Number of data bits
 * @param polynomial CRC generator polynomial
 * @return OBDD representing CRC calculation constraint
 */
OBDD* obdd_crc_polynomial(int bits, int polynomial);

/**
 * @brief Construct BDD for elliptic curve point addition
 * 
 * @param bits Number of bits for coordinates
 * @param a Elliptic curve parameter a
 * @param b Elliptic curve parameter b  
 * @param modulus Prime modulus p
 * @return OBDD representing point addition on elliptic curve
 */
OBDD* obdd_ecc_point_addition(int bits, int a, int b, int modulus);

/* --------------------------------------------------------------------------
 * DIOPHANTINE EQUATIONS AND ALGEBRAIC CONSTRAINTS
 * 
 * Implementation of classical Diophantine equations and algebraic constraint
 * systems using BDD representation for systematic solution exploration.
 * -------------------------------------------------------------------------- */

/**
 * @brief Construct BDD for linear Diophantine equation: ax + by = c
 * 
 * @param bits Number of bits per variable
 * @param a Coefficient of x
 * @param b Coefficient of y  
 * @param c Constant term
 * @return OBDD representing all integer solutions within bit range
 */
OBDD* obdd_linear_diophantine(int bits, int a, int b, int c);

/**
 * @brief Construct BDD for Pell equation: x² - Dy² = 1
 * 
 * @param bits Number of bits per variable
 * @param D Non-square constant
 * @return OBDD representing solutions to Pell equation
 */
OBDD* obdd_pell_equation(int bits, int D);

/**
 * @brief Construct BDD for Pythagorean triples: x² + y² = z²
 * 
 * @param bits Number of bits per variable
 * @return OBDD representing Pythagorean triples
 */
OBDD* obdd_pythagorean_triples(int bits);

/* --------------------------------------------------------------------------
 * COMBINATORIAL OPTIMIZATION PROBLEMS
 * 
 * Classical NP-hard combinatorial problems implemented as BDD constraint
 * systems for exact solution enumeration and optimization analysis.
 * -------------------------------------------------------------------------- */

/**
 * @brief Construct BDD for N-Queens constraint satisfaction problem
 * 
 * @param n Board size (n x n chessboard)
 * @return OBDD with n² variables representing valid queen placements
 */
OBDD* obdd_n_queens(int n);

/**
 * @brief Construct BDD for Graph 3-coloring problem
 * 
 * @param num_vertices Number of vertices in the graph
 * @param edges Array of vertex pairs representing graph edges
 * @param num_edges Number of edges in the graph
 * @return OBDD representing valid 3-colorings
 */
OBDD* obdd_graph_3_coloring(int num_vertices, int (*edges)[2], int num_edges);

/**
 * @brief Construct BDD for Hamiltonian path problem
 * 
 * @param num_vertices Number of vertices in the graph
 * @param adjacency_matrix Adjacency matrix (flattened array)
 * @return OBDD representing valid Hamiltonian paths
 */
OBDD* obdd_hamiltonian_path(int num_vertices, int* adjacency_matrix);

/**
 * @brief Construct BDD for 0-1 Knapsack optimization problem
 * 
 * @param num_items Number of items to choose from
 * @param weights Array of item weights
 * @param values Array of item values
 * @param capacity Knapsack weight capacity
 * @param min_value Minimum value requirement for valid solutions
 * @return OBDD representing valid knapsack solutions
 */
OBDD* obdd_knapsack(int num_items, int* weights, int* values, int capacity, int min_value);

/* --------------------------------------------------------------------------
 * BOOLEAN SATISFIABILITY AND CONSTRAINT SATISFACTION
 * 
 * SAT and CSP problem implementations demonstrating BDD effectiveness
 * for constraint satisfaction and combinatorial search problems.
 * -------------------------------------------------------------------------- */

/**
 * @brief Construct BDD from Conjunctive Normal Form (CNF) formula
 * 
 * @param num_vars Number of Boolean variables
 * @param clauses Array of clauses (each clause is array of literals, 0-terminated)
 * @param num_clauses Number of clauses in the formula
 * @return OBDD representing satisfying assignments
 */
OBDD* obdd_from_cnf(int num_vars, int** clauses, int num_clauses);

/**
 * @brief Construct BDD for random 3-SAT instance
 * 
 * @param num_vars Number of Boolean variables
 * @param num_clauses Number of clauses to generate
 * @param seed Random number generator seed for reproducibility
 * @return OBDD representing the randomly generated 3-SAT instance
 */
OBDD* obdd_random_3sat(int num_vars, int num_clauses, unsigned int seed);

/**
 * @brief Construct BDD for Sudoku puzzle constraint system
 * 
 * @param puzzle 9x9 array with given clues (0 for empty cells, 1-9 for given digits)
 * @return OBDD representing valid Sudoku completions
 */
OBDD* obdd_sudoku(int puzzle[9][9]);

/* --------------------------------------------------------------------------
 * SOLUTION ANALYSIS AND BENCHMARKING UTILITIES
 * 
 * Advanced analysis functions for mathematical constraint problems including
 * solution counting, enumeration, and comprehensive benchmarking capabilities.
 * -------------------------------------------------------------------------- */

/**
 * @brief Count satisfying assignments in OBDD
 * 
 * @param bdd The BDD to analyze
 * @return Number of satisfying assignments (limited to 64-bit precision)
 * 
 * SOLUTION COUNTING:
 * Implements efficient BDD-based solution counting for constraint satisfaction
 * problems. Uses dynamic programming on the BDD structure for optimal performance.
 */
uint64_t obdd_count_solutions(const OBDD* bdd);

/**
 * @brief Enumerate satisfying assignments from OBDD
 * 
 * @param bdd The BDD to enumerate solutions from
 * @param assignments Output array (caller must allocate)
 * @param max_assignments Maximum number of assignments to enumerate
 * @return Actual number of assignments found and stored
 * 
 * SOLUTION ENUMERATION:
 * Systematic enumeration of satisfying assignments using BDD traversal
 * algorithms. Enables interactive exploration of constraint solution spaces.
 */
int obdd_enumerate_solutions(const OBDD* bdd, int** assignments, int max_assignments);

/**
 * @brief Comprehensive benchmark result structure for mathematical problems
 * 
 * BENCHMARK METRICS:
 * This structure captures detailed performance metrics for mathematical
 * constraint problems, enabling systematic evaluation of BDD effectiveness
 * across different mathematical domains.
 */
typedef struct {
    const char* problem_name;           /**< Descriptive name of the mathematical problem */
    int problem_size;                   /**< Problem instance size parameter */
    int num_variables;                  /**< Number of Boolean variables in BDD encoding */
    int bdd_size;                       /**< Number of nodes in the constructed BDD */
    double construction_time_ms;        /**< Time required for BDD construction in milliseconds */
    uint64_t num_solutions;             /**< Number of satisfying assignments found */
    double reordering_time_ms;          /**< Time required for variable reordering optimization */
    int optimized_bdd_size;            /**< BDD size after variable reordering optimization */
    double reduction_ratio;             /**< Size reduction ratio achieved through optimization */
} AdvancedBenchmark;

/**
 * @brief Execute comprehensive benchmark suite for mathematical problems
 * 
 * @param results Output array for storing benchmark results
 * @param max_results Maximum number of benchmark results to store
 * @return Number of benchmarks successfully completed
 * 
 * COMPREHENSIVE BENCHMARKING:
 * Executes systematic benchmarking across all implemented mathematical
 * constraint problems, providing detailed performance analysis and
 * optimization effectiveness evaluation.
 */
int obdd_run_advanced_benchmarks(AdvancedBenchmark* results, int max_results);

/**
 * @brief Print benchmark results in formatted table
 * 
 * @param results Array of benchmark results to display
 * @param num_results Number of results in the array
 * 
 * RESULTS VISUALIZATION:
 * Formats and displays benchmark results in human-readable tabular format
 * for analysis and documentation purposes.
 */
void obdd_print_benchmark_results(const AdvancedBenchmark* results, int num_results);

#ifdef __cplusplus
}
#endif

#endif /* OBDD_ADVANCED_MATH_HPP */