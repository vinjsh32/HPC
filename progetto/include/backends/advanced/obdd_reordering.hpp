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
 * @file obdd_reordering.hpp
 * @brief Variable Reordering Optimization Algorithms
 * 
 * This file is part of the high-performance OBDD library providing
 * comprehensive Binary Decision Diagram operations with multi-backend
 * support for Sequential CPU, OpenMP Parallel, and CUDA GPU execution.
 * 
 * @author @vijsh32
 * @date August 25, 2024
 * @version 2.1
 * @copyright 2024 High Performance Computing Laboratory
 */


#pragma once
#ifndef OBDD_REORDERING_HPP
#define OBDD_REORDERING_HPP

/**
 * @file obdd_reordering.hpp
 * @brief Advanced variable reordering algorithms for OBDD optimization
 * 
 * This module implements state-of-the-art algorithms for finding optimal
 * or near-optimal variable orderings to minimize BDD size:
 * 
 * - Sifting Algorithm: Moves variables to locally optimal positions
 * - Window Permutation: Exhaustive search within small windows using DP
 * - Simulated Annealing: Global optimization with probabilistic moves
 * - Genetic Algorithm: Evolutionary approach for large variable sets
 */

#include "obdd.hpp"
#include <vector>
#include <functional>
#include <random>

#ifdef __cplusplus
extern "C" {
#endif

/* =====================================================
   REORDERING STRATEGY SELECTION
   ===================================================== */

typedef enum {
    REORDER_SIFTING = 0,
    REORDER_WINDOW_DP = 1, 
    REORDER_SIMULATED_ANNEALING = 2,
    REORDER_GENETIC = 3,
    REORDER_HYBRID = 4  // Combines multiple strategies
} ReorderStrategy;

typedef struct {
    ReorderStrategy strategy;
    int max_iterations;
    double temperature_initial;  // For simulated annealing
    double cooling_rate;         // For simulated annealing
    int window_size;            // For window permutation
    int population_size;        // For genetic algorithm
    int tournament_size;        // For genetic algorithm
    double mutation_rate;       // For genetic algorithm
    double crossover_rate;      // For genetic algorithm
    bool enable_parallel;       // Use OpenMP when available
} ReorderConfig;

/* =====================================================
   REORDERING METRICS AND EVALUATION
   ===================================================== */

typedef struct {
    int initial_size;
    int final_size;
    double reduction_ratio;
    double execution_time_ms;
    int iterations_performed;
    int swaps_performed;
    const char* algorithm_used;
} ReorderResult;

/* Function pointer for custom BDD size evaluation */
typedef int (*BDDSizeEvaluator)(const OBDD* bdd);

/* =====================================================
   MAIN REORDERING API
   ===================================================== */

/**
 * @brief Apply advanced reordering algorithm to minimize BDD size
 * @param bdd The BDD to reorder (modified in-place)
 * @param config Configuration parameters for the algorithm
 * @param result Output statistics (can be NULL)
 * @return New optimized variable ordering (caller must free)
 */
int* obdd_reorder_advanced(OBDD* bdd, 
                          const ReorderConfig* config, 
                          ReorderResult* result);

/**
 * @brief Get default configuration for a reordering strategy
 */
ReorderConfig obdd_reorder_get_default_config(ReorderStrategy strategy);

/* =====================================================
   INDIVIDUAL ALGORITHM IMPLEMENTATIONS
   ===================================================== */

/**
 * @brief Sifting algorithm - move each variable to optimal position
 * 
 * The sifting algorithm works by:
 * 1. For each variable v, temporarily move it through all positions
 * 2. Find the position that minimizes total BDD size
 * 3. Place v at that optimal position
 * 4. Repeat until no improvement is found
 */
int* obdd_reorder_sifting(OBDD* bdd, 
                         int max_iterations,
                         BDDSizeEvaluator size_eval,
                         ReorderResult* result);

/**
 * @brief Window permutation with dynamic programming
 * 
 * Exhaustively tries all permutations within sliding windows:
 * 1. Slide a window of size k across variable ordering
 * 2. For each window position, try all k! permutations
 * 3. Use DP to avoid recomputing identical subproblems
 * 4. Select the permutation that minimizes BDD size
 */
int* obdd_reorder_window_dp(OBDD* bdd,
                           int window_size,
                           BDDSizeEvaluator size_eval,
                           ReorderResult* result);

/**
 * @brief Simulated annealing global optimization
 * 
 * Probabilistic algorithm that accepts worse solutions with decreasing probability:
 * 1. Start with random ordering and high temperature
 * 2. Generate neighbor by swapping adjacent variables
 * 3. Accept improvement always, worse solutions with probability exp(-ΔE/T)
 * 4. Gradually cool temperature according to cooling schedule
 */
int* obdd_reorder_simulated_annealing(OBDD* bdd,
                                     double initial_temp,
                                     double cooling_rate,
                                     int max_iterations,
                                     BDDSizeEvaluator size_eval,
                                     ReorderResult* result);

/**
 * @brief Genetic algorithm for variable ordering
 * 
 * Evolutionary approach for large variable sets:
 * 1. Initialize random population of variable orderings
 * 2. Evaluate fitness (inverse of BDD size)
 * 3. Select parents using tournament selection
 * 4. Create offspring using order-preserving crossover
 * 5. Apply mutation (random swaps)
 * 6. Replace worst individuals with offspring
 */
int* obdd_reorder_genetic(OBDD* bdd,
                         int population_size,
                         int generations,
                         double mutation_rate,
                         double crossover_rate,
                         BDDSizeEvaluator size_eval,
                         ReorderResult* result);

/* =====================================================
   PARALLEL VERSIONS (OpenMP)
   ===================================================== */

#ifdef OBDD_ENABLE_OPENMP

int* obdd_reorder_sifting_omp(OBDD* bdd,
                             int max_iterations,
                             BDDSizeEvaluator size_eval,
                             ReorderResult* result);

int* obdd_reorder_genetic_omp(OBDD* bdd,
                             int population_size,
                             int generations,
                             double mutation_rate,
                             double crossover_rate,
                             BDDSizeEvaluator size_eval,
                             ReorderResult* result);

#endif /* OBDD_ENABLE_OPENMP */

/* =====================================================
   UTILITY FUNCTIONS
   ===================================================== */

/**
 * @brief Default BDD size evaluator (counts total nodes)
 */
int obdd_count_nodes(const OBDD* bdd);

/**
 * @brief Memory-aware size evaluator (considers memory layout)
 */
int obdd_count_memory_footprint(const OBDD* bdd);

/**
 * @brief Swap two adjacent variables in the ordering
 * @param bdd The BDD to modify
 * @param pos Position of first variable (0-based)
 * @return New BDD size after swap, -1 on error
 */
int obdd_swap_adjacent_variables(OBDD* bdd, int pos);

/**
 * @brief Apply a complete variable ordering to a BDD
 * @param bdd The BDD to reorder
 * @param new_order Array of variable indices
 * @param size Number of variables
 * @return 0 on success, -1 on error
 */
int obdd_apply_variable_order(OBDD* bdd, const int* new_order, int size);

/**
 * @brief Generate random variable ordering (for testing/initialization)
 */
int* obdd_generate_random_ordering(int num_vars, unsigned int seed);

/**
 * @brief Pretty print reordering results
 */
void obdd_print_reorder_result(const ReorderResult* result);

#ifdef __cplusplus
}
#endif

#endif /* OBDD_REORDERING_HPP */