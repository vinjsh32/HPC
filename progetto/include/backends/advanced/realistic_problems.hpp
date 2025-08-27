/**
 * @file realistic_problems.hpp
 * @brief Realistic large-scale problem generators for OBDD benchmarking
 * @version 2.0
 * @date 2024
 * 
 * This module generates challenging, real-world problem instances for comprehensive
 * OBDD performance evaluation including:
 * - Large-scale cryptographic functions (AES, SHA, RSA components)
 * - Complex combinatorial problems (N-Queens, Graph Coloring, SAT)
 * - Industrial verification benchmarks (circuit verification, model checking)
 * - Mathematical constraint problems (Sudoku variants, optimization)
 * 
 * @author HPC Team
 * @copyright 2024 High Performance Computing Laboratory
 */

#pragma once

#include "core/obdd.hpp"
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Problem categories for realistic benchmarking
 */
typedef enum {
    PROBLEM_CRYPTOGRAPHIC = 0,      // AES, SHA, DES components
    PROBLEM_COMBINATORIAL,          // N-Queens, Graph Coloring
    PROBLEM_VERIFICATION,           // Circuit verification, model checking
    PROBLEM_MATHEMATICAL,           // Sudoku, constraint satisfaction
    PROBLEM_SAT_INSTANCES,          // Boolean satisfiability problems
    PROBLEM_INDUSTRIAL,             // Real-world industrial benchmarks
    PROBLEM_STRESS_TEST             // Maximum difficulty stress tests
} ProblemCategory;

/**
 * @brief Problem complexity levels
 */
typedef enum {
    COMPLEXITY_SMALL = 1,       // 10-15 variables, testing overhead
    COMPLEXITY_MEDIUM,          // 16-20 variables, typical problems
    COMPLEXITY_LARGE,           // 21-25 variables, challenging problems
    COMPLEXITY_HUGE,            // 26-30 variables, stress testing
    COMPLEXITY_EXTREME          // 30+ variables, maximum challenge
} ProblemComplexity;

/**
 * @brief Problem instance descriptor
 */
typedef struct {
    ProblemCategory category;
    ProblemComplexity complexity;
    char name[128];
    char description[512];
    
    int num_variables;
    int expected_nodes;         // Expected BDD size
    double expected_time_ms;    // Expected CPU time
    
    // Problem-specific parameters
    union {
        struct {
            int key_bits;
            int rounds;
        } crypto;
        
        struct {
            int board_size;
            int num_queens;
        } nqueens;
        
        struct {
            int num_vertices;
            int num_colors;
            int edge_density_percent;
        } graph_coloring;
        
        struct {
            int grid_size;
            int num_clues;
            int variant_type;
        } sudoku;
        
        struct {
            int num_clauses;
            int clause_length;
            double satisfiability_ratio;
        } sat;
        
        struct {
            int circuit_inputs;
            int circuit_gates;
            int circuit_depth;
        } verification;
    } params;
    
} RealisticProblem;

/**
 * @brief Benchmark suite configuration for realistic problems
 */
typedef struct {
    ProblemComplexity min_complexity;
    ProblemComplexity max_complexity;
    
    int include_cryptographic;
    int include_combinatorial;
    int include_verification;
    int include_mathematical;
    int include_sat;
    int include_industrial;
    
    int problems_per_category;
    int repetitions_per_problem;
    
    double timeout_seconds;
    size_t memory_limit_bytes;
    
} RealisticBenchmarkConfig;

// =====================================================
// Problem Generators - Cryptographic
// =====================================================

/**
 * @brief Generate AES S-box transformation as OBDD
 * @param key_bits AES key size (128, 192, 256)
 * @param num_rounds Number of AES rounds to model
 * @return Generated OBDD representing AES transformation
 */
OBDD* realistic_generate_aes_sbox(int key_bits, int num_rounds);

/**
 * @brief Generate AES MixColumns transformation
 * @param complexity Problem complexity level
 * @return Generated OBDD for AES MixColumns
 */
OBDD* realistic_generate_aes_mixcolumns(ProblemComplexity complexity);

/**
 * @brief Generate SHA-256 compression function component
 * @param complexity Problem complexity level
 * @return Generated OBDD for SHA component
 */
OBDD* realistic_generate_sha256_component(ProblemComplexity complexity);

/**
 * @brief Generate RSA modular exponentiation component
 * @param bit_width RSA key bit width
 * @param exponent_bits Exponent bit width
 * @return Generated OBDD for RSA component
 */
OBDD* realistic_generate_rsa_component(int bit_width, int exponent_bits);

/**
 * @brief Generate elliptic curve point addition
 * @param field_bits Finite field bit width
 * @param curve_type Elliptic curve type (0=secp256k1, 1=P-256)
 * @return Generated OBDD for ECC point addition
 */
OBDD* realistic_generate_ecc_point_addition(int field_bits, int curve_type);

// =====================================================
// Problem Generators - Combinatorial
// =====================================================

/**
 * @brief Generate large N-Queens problem
 * @param board_size Size of chessboard (8-16 for realistic problems)
 * @param constraint_type Type of constraints (0=basic, 1=optimized)
 * @return Generated OBDD representing N-Queens constraints
 */
OBDD* realistic_generate_nqueens_large(int board_size, int constraint_type);

/**
 * @brief Generate graph coloring problem
 * @param num_vertices Number of graph vertices
 * @param num_colors Number of colors available
 * @param edge_density_percent Edge density (10-90%)
 * @return Generated OBDD for graph coloring constraints
 */
OBDD* realistic_generate_graph_coloring(int num_vertices, int num_colors, 
                                       int edge_density_percent);

/**
 * @brief Generate Hamiltonian path problem
 * @param num_vertices Number of graph vertices
 * @param graph_type Graph type (0=random, 1=grid, 2=complete)
 * @return Generated OBDD for Hamiltonian path constraints
 */
OBDD* realistic_generate_hamiltonian_path(int num_vertices, int graph_type);

/**
 * @brief Generate knapsack problem with multiple constraints
 * @param num_items Number of items
 * @param num_knapsacks Number of knapsacks (multi-dimensional)
 * @param complexity Problem complexity level
 * @return Generated OBDD for knapsack constraints
 */
OBDD* realistic_generate_multi_knapsack(int num_items, int num_knapsacks,
                                       ProblemComplexity complexity);

/**
 * @brief Generate traveling salesman problem (TSP)
 * @param num_cities Number of cities
 * @param distance_type Distance metric (0=Euclidean, 1=Manhattan)
 * @return Generated OBDD for TSP constraints
 */
OBDD* realistic_generate_tsp(int num_cities, int distance_type);

// =====================================================
// Problem Generators - Verification
// =====================================================

/**
 * @brief Generate circuit equivalence checking problem
 * @param circuit_inputs Number of circuit inputs
 * @param circuit_gates Number of gates
 * @param circuit_depth Logic depth
 * @return Generated OBDD for circuit verification
 */
OBDD* realistic_generate_circuit_verification(int circuit_inputs, int circuit_gates, 
                                             int circuit_depth);

/**
 * @brief Generate model checking problem (safety property)
 * @param state_variables Number of state variables
 * @param transition_complexity Transition system complexity
 * @return Generated OBDD for model checking
 */
OBDD* realistic_generate_model_checking(int state_variables, int transition_complexity);

/**
 * @brief Generate hardware multiplier verification
 * @param multiplier_bits Bit width of multiplier
 * @param algorithm_type Multiplication algorithm (0=booth, 1=wallace)
 * @return Generated OBDD for multiplier verification
 */
OBDD* realistic_generate_multiplier_verification(int multiplier_bits, int algorithm_type);

/**
 * @brief Generate cache coherence protocol verification
 * @param num_processors Number of processors
 * @param cache_levels Number of cache levels
 * @return Generated OBDD for cache coherence verification
 */
OBDD* realistic_generate_cache_coherence(int num_processors, int cache_levels);

// =====================================================
// Problem Generators - Mathematical
// =====================================================

/**
 * @brief Generate large Sudoku variant
 * @param grid_size Sudoku grid size (9, 16, 25)
 * @param variant_type Sudoku variant (0=classic, 1=diagonal, 2=irregular)
 * @param num_clues Number of initial clues
 * @return Generated OBDD for Sudoku constraints
 */
OBDD* realistic_generate_sudoku_large(int grid_size, int variant_type, int num_clues);

/**
 * @brief Generate Latin square problem
 * @param square_size Size of Latin square
 * @param orthogonal_pairs Number of orthogonal pairs required
 * @return Generated OBDD for Latin square constraints
 */
OBDD* realistic_generate_latin_square(int square_size, int orthogonal_pairs);

/**
 * @brief Generate magic square problem
 * @param square_size Size of magic square
 * @param magic_constant Target magic constant
 * @return Generated OBDD for magic square constraints
 */
OBDD* realistic_generate_magic_square(int square_size, int magic_constant);

/**
 * @brief Generate bin packing problem
 * @param num_items Number of items to pack
 * @param num_bins Number of bins available
 * @param bin_capacity Capacity of each bin
 * @return Generated OBDD for bin packing constraints
 */
OBDD* realistic_generate_bin_packing(int num_items, int num_bins, int bin_capacity);

// =====================================================
// Problem Generators - SAT Instances
// =====================================================

/**
 * @brief Generate random 3-SAT instance
 * @param num_variables Number of boolean variables
 * @param num_clauses Number of clauses
 * @param satisfiability_ratio Expected satisfiability ratio (0.0-1.0)
 * @return Generated OBDD for 3-SAT instance
 */
OBDD* realistic_generate_3sat_random(int num_variables, int num_clauses, 
                                    double satisfiability_ratio);

/**
 * @brief Generate k-SAT instance with structured patterns
 * @param num_variables Number of boolean variables
 * @param k Clause length
 * @param structure_type Structure type (0=random, 1=grid, 2=hierarchical)
 * @return Generated OBDD for k-SAT instance
 */
OBDD* realistic_generate_ksat_structured(int num_variables, int k, int structure_type);

/**
 * @brief Generate MAX-SAT instance
 * @param num_variables Number of boolean variables
 * @param num_hard_clauses Number of hard clauses
 * @param num_soft_clauses Number of soft clauses
 * @return Generated OBDD for MAX-SAT instance
 */
OBDD* realistic_generate_maxsat(int num_variables, int num_hard_clauses, 
                              int num_soft_clauses);

// =====================================================
// Problem Suite Management
// =====================================================

/**
 * @brief Get default configuration for realistic benchmarks
 * @return Default benchmark configuration
 */
RealisticBenchmarkConfig realistic_get_default_config(void);

/**
 * @brief Generate comprehensive problem suite
 * @param config Benchmark configuration
 * @param problems Output array for generated problems
 * @param max_problems Maximum number of problems to generate
 * @return Number of problems generated
 */
int realistic_generate_problem_suite(const RealisticBenchmarkConfig* config,
                                   RealisticProblem* problems, int max_problems);

/**
 * @brief Create OBDD for specific problem instance
 * @param problem Problem descriptor
 * @return Generated OBDD, NULL on failure
 */
OBDD* realistic_create_problem_obdd(const RealisticProblem* problem);

/**
 * @brief Estimate problem difficulty
 * @param problem Problem descriptor
 * @param estimated_nodes Output: estimated BDD nodes
 * @param estimated_time_ms Output: estimated computation time
 * @return 0 on success, -1 on failure
 */
int realistic_estimate_difficulty(const RealisticProblem* problem, 
                                int* estimated_nodes, double* estimated_time_ms);

/**
 * @brief Validate problem solution
 * @param problem Problem descriptor
 * @param solution_bdd Solution OBDD to validate
 * @param validation_result Output: validation result description
 * @param buffer_size Size of validation result buffer
 * @return 1 if valid, 0 if invalid, -1 on error
 */
int realistic_validate_solution(const RealisticProblem* problem, const OBDD* solution_bdd,
                              char* validation_result, size_t buffer_size);

// =====================================================
// Benchmark Analysis
// =====================================================

/**
 * @brief Analyze problem characteristics
 * @param problem Problem descriptor
 * @param analysis_output Output buffer for analysis
 * @param buffer_size Size of analysis buffer
 */
void realistic_analyze_problem(const RealisticProblem* problem, 
                             char* analysis_output, size_t buffer_size);

/**
 * @brief Compare problem difficulties
 * @param problems Array of problem descriptors
 * @param num_problems Number of problems
 * @param comparison_output Output buffer for comparison
 * @param buffer_size Size of comparison buffer
 */
void realistic_compare_difficulties(const RealisticProblem* problems, int num_problems,
                                  char* comparison_output, size_t buffer_size);

/**
 * @brief Generate problem statistics
 * @param problems Array of problem descriptors
 * @param num_problems Number of problems
 * @param stats_output Output buffer for statistics
 * @param buffer_size Size of statistics buffer
 */
void realistic_generate_statistics(const RealisticProblem* problems, int num_problems,
                                 char* stats_output, size_t buffer_size);

// =====================================================
// Utility Functions
// =====================================================

/**
 * @brief Get problem category name
 * @param category Problem category
 * @return Category name string
 */
const char* realistic_get_category_name(ProblemCategory category);

/**
 * @brief Get complexity level name
 * @param complexity Complexity level
 * @return Complexity name string
 */
const char* realistic_get_complexity_name(ProblemComplexity complexity);

/**
 * @brief Print problem descriptor
 * @param problem Problem descriptor to print
 */
void realistic_print_problem(const RealisticProblem* problem);

#ifdef __cplusplus
}
#endif