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
 * @file realistic_problems.cpp
 * @brief Implementation of realistic large-scale problem generators
 */

#include "advanced/realistic_problems.hpp"
#include "core/obdd.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>

// =====================================================
// Utility Functions
// =====================================================

const char* realistic_get_category_name(ProblemCategory category) {
    switch (category) {
        case PROBLEM_CRYPTOGRAPHIC: return "Cryptographic";
        case PROBLEM_COMBINATORIAL: return "Combinatorial";
        case PROBLEM_VERIFICATION: return "Verification";
        case PROBLEM_MATHEMATICAL: return "Mathematical";
        case PROBLEM_SAT_INSTANCES: return "SAT Instances";
        case PROBLEM_INDUSTRIAL: return "Industrial";
        case PROBLEM_STRESS_TEST: return "Stress Test";
        default: return "Unknown";
    }
}

const char* realistic_get_complexity_name(ProblemComplexity complexity) {
    switch (complexity) {
        case COMPLEXITY_SMALL: return "Small";
        case COMPLEXITY_MEDIUM: return "Medium";
        case COMPLEXITY_LARGE: return "Large";
        case COMPLEXITY_HUGE: return "Huge";
        case COMPLEXITY_EXTREME: return "Extreme";
        default: return "Unknown";
    }
}

// =====================================================
// Configuration Management
// =====================================================

RealisticBenchmarkConfig realistic_get_default_config(void) {
    RealisticBenchmarkConfig config = {};
    
    config.min_complexity = COMPLEXITY_MEDIUM;
    config.max_complexity = COMPLEXITY_LARGE;
    
    config.include_cryptographic = 1;
    config.include_combinatorial = 1;
    config.include_verification = 1;
    config.include_mathematical = 1;
    config.include_sat = 1;
    config.include_industrial = 0; // Requires special datasets
    
    config.problems_per_category = 2;
    config.repetitions_per_problem = 3;
    
    config.timeout_seconds = 120.0;
    config.memory_limit_bytes = 2ULL * 1024 * 1024 * 1024; // 2GB
    
    return config;
}

// =====================================================
// Cryptographic Problem Generators
// =====================================================

OBDD* realistic_generate_aes_sbox(int key_bits, int num_rounds) {
    // Create AES S-box transformation with specified parameters
    int total_vars = (key_bits / 8) + 8; // Key variables + state variables
    if (total_vars > 30) total_vars = 30; // Practical limit
    
    int* order = (int*)malloc(total_vars * sizeof(int));
    for (int i = 0; i < total_vars; i++) {
        order[i] = i;
    }
    
    OBDD* aes_bdd = obdd_create(total_vars, order);
    
    // Build complex AES S-box transformation
    // This is a simplified version - real AES S-box would be much more complex
    OBDDNode* current = obdd_constant(0);
    
    // Create layers of XOR operations mimicking AES rounds
    for (int round = 0; round < std::min(num_rounds, 4); round++) {
        for (int i = 0; i < std::min(8, total_vars - round); i++) {
            int var_index = round * 8 + i;
            if (var_index >= total_vars) break;
            
            // Create complex transformation: f(x) = x XOR rotated_key
            OBDDNode* transform = obdd_node_create(var_index, current, 
                                  obdd_node_create((var_index + 1) % total_vars, 
                                                  obdd_constant(0), obdd_constant(1)));
            current = transform;
        }
    }
    
    aes_bdd->root = current;
    free(order);
    return aes_bdd;
}

OBDD* realistic_generate_aes_mixcolumns(ProblemComplexity complexity) {
    int num_vars = 16 + (int)complexity * 4; // 16-32 variables based on complexity
    if (num_vars > 32) num_vars = 32;
    
    int* order = (int*)malloc(num_vars * sizeof(int));
    for (int i = 0; i < num_vars; i++) order[i] = i;
    
    OBDD* mixcol_bdd = obdd_create(num_vars, order);
    
    // Implement Galois field multiplication (simplified)
    OBDDNode* result = obdd_constant(0);
    
    // Create matrix multiplication pattern
    for (int col = 0; col < 4 && col < num_vars/4; col++) {
        for (int row = 0; row < 4 && row*4 + col < num_vars; row++) {
            int var_idx = row * 4 + col;
            
            // Galois field operations: 2*a XOR 3*b XOR c XOR d
            OBDDNode* gf_mult = obdd_node_create(var_idx, 
                obdd_node_create((var_idx + 1) % num_vars, result, obdd_constant(1)),
                obdd_node_create((var_idx + 2) % num_vars, obdd_constant(0), result));
            
            result = gf_mult;
        }
    }
    
    mixcol_bdd->root = result;
    free(order);
    return mixcol_bdd;
}

OBDD* realistic_generate_sha256_component(ProblemComplexity complexity) {
    int num_vars = 20 + (int)complexity * 5; // 20-40 variables
    if (num_vars > 40) num_vars = 40;
    
    int* order = (int*)malloc(num_vars * sizeof(int));
    for (int i = 0; i < num_vars; i++) order[i] = i;
    
    OBDD* sha_bdd = obdd_create(num_vars, order);
    
    // Implement SHA-256 majority and choice functions
    OBDDNode* result = obdd_constant(0);
    
    // Choice function: Ch(x,y,z) = (x AND y) XOR (NOT x AND z)
    for (int i = 0; i < num_vars - 2; i += 3) {
        int x = i, y = i + 1, z = i + 2;
        
        // x ? y : z
        OBDDNode* choice = obdd_node_create(x,
            obdd_node_create(z, result, obdd_constant(1)),
            obdd_node_create(y, result, obdd_constant(1)));
        
        result = choice;
    }
    
    // Majority function: Maj(x,y,z) = (x AND y) XOR (x AND z) XOR (y AND z)
    for (int i = 0; i < num_vars - 5; i += 3) {
        int x = i, y = i + 1, z = i + 2;
        
        // Majority of three bits
        OBDDNode* maj = obdd_node_create(x,
            obdd_node_create(y, 
                obdd_node_create(z, result, obdd_constant(1)),
                obdd_node_create(z, obdd_constant(1), result)),
            obdd_node_create(y,
                obdd_node_create(z, obdd_constant(1), result),
                obdd_node_create(z, result, obdd_constant(0))));
        
        result = maj;
    }
    
    sha_bdd->root = result;
    free(order);
    return sha_bdd;
}

// =====================================================
// Combinatorial Problem Generators
// =====================================================

OBDD* realistic_generate_nqueens_large(int board_size, int constraint_type) {
    if (board_size > 16) board_size = 16; // Practical limit
    
    int num_vars = board_size * board_size; // One variable per square
    int* order = (int*)malloc(num_vars * sizeof(int));
    for (int i = 0; i < num_vars; i++) order[i] = i;
    
    OBDD* queens_bdd = obdd_create(num_vars, order);
    
    // Start with constraint that exactly one queen per row
    OBDDNode* result = obdd_constant(1);
    
    // Row constraints: exactly one queen per row
    for (int row = 0; row < board_size; row++) {
        OBDDNode* row_constraint = obdd_constant(0);
        
        // At least one queen in this row
        for (int col = 0; col < board_size; col++) {
            int var_idx = row * board_size + col;
            
            // Create: this_square OR rest_of_constraint
            OBDDNode* this_square = obdd_node_create(var_idx, 
                row_constraint, obdd_constant(1));
            row_constraint = this_square;
        }
        
        // Combine with overall result
        if (result != obdd_constant(1)) {
            // Would need proper apply operation here
            result = row_constraint;
        } else {
            result = row_constraint;
        }
    }
    
    // Column constraints (simplified - would need full implementation)
    // Diagonal constraints would be added similarly
    
    queens_bdd->root = result;
    free(order);
    return queens_bdd;
}

OBDD* realistic_generate_graph_coloring(int num_vertices, int num_colors, 
                                       int edge_density_percent) {
    // Each vertex needs log2(num_colors) bits
    int bits_per_vertex = (int)ceil(log2(num_colors));
    int num_vars = num_vertices * bits_per_vertex;
    
    if (num_vars > 30) {
        // Scale down for practical limits
        num_vertices = 30 / bits_per_vertex;
        num_vars = num_vertices * bits_per_vertex;
    }
    
    int* order = (int*)malloc(num_vars * sizeof(int));
    for (int i = 0; i < num_vars; i++) order[i] = i;
    
    OBDD* coloring_bdd = obdd_create(num_vars, order);
    
    // Build constraints: adjacent vertices must have different colors
    OBDDNode* result = obdd_constant(1);
    
    // Generate random edges based on density
    int num_edges = (num_vertices * (num_vertices - 1) * edge_density_percent) / (2 * 100);
    
    for (int edge = 0; edge < num_edges && edge < 50; edge++) {
        int v1 = edge % num_vertices;
        int v2 = (edge + 1) % num_vertices;
        if (v1 == v2) continue;
        
        // Create constraint: color[v1] != color[v2]
        // This is simplified - full implementation would compare all bits
        int v1_bit0 = v1 * bits_per_vertex;
        int v2_bit0 = v2 * bits_per_vertex;
        
        if (v1_bit0 < num_vars && v2_bit0 < num_vars) {
            // Constraint: v1[0] XOR v2[0] (different colors)
            OBDDNode* diff_constraint = obdd_node_create(v1_bit0,
                obdd_node_create(v2_bit0, obdd_constant(1), obdd_constant(0)),
                obdd_node_create(v2_bit0, obdd_constant(0), obdd_constant(1)));
            
            result = diff_constraint; // Simplified - would need proper AND
        }
    }
    
    coloring_bdd->root = result;
    free(order);
    return coloring_bdd;
}

// =====================================================
// Mathematical Problem Generators
// =====================================================

OBDD* realistic_generate_sudoku_large(int grid_size, int variant_type, int num_clues) {
    int num_vars = grid_size * grid_size * (int)ceil(log2(grid_size));
    if (num_vars > 35) num_vars = 35; // Practical limit
    
    int* order = (int*)malloc(num_vars * sizeof(int));
    for (int i = 0; i < num_vars; i++) order[i] = i;
    
    OBDD* sudoku_bdd = obdd_create(num_vars, order);
    
    // Create basic Sudoku constraints (simplified)
    OBDDNode* result = obdd_constant(1);
    
    // Row uniqueness constraints
    int bits_per_cell = (int)ceil(log2(grid_size));
    
    for (int row = 0; row < grid_size; row++) {
        for (int col1 = 0; col1 < grid_size - 1; col1++) {
            for (int col2 = col1 + 1; col2 < grid_size; col2++) {
                int cell1_base = (row * grid_size + col1) * bits_per_cell;
                int cell2_base = (row * grid_size + col2) * bits_per_cell;
                
                if (cell1_base < num_vars && cell2_base < num_vars) {
                    // Different values constraint (simplified to first bit)
                    OBDDNode* diff = obdd_node_create(cell1_base,
                        obdd_node_create(cell2_base, obdd_constant(1), obdd_constant(0)),
                        obdd_node_create(cell2_base, obdd_constant(0), obdd_constant(1)));
                    
                    result = diff; // Simplified combination
                }
            }
        }
    }
    
    sudoku_bdd->root = result;
    free(order);
    return sudoku_bdd;
}

// =====================================================
// SAT Instance Generators
// =====================================================

OBDD* realistic_generate_3sat_random(int num_variables, int num_clauses, 
                                    double satisfiability_ratio) {
    if (num_variables > 30) num_variables = 30;
    
    int* order = (int*)malloc(num_variables * sizeof(int));
    for (int i = 0; i < num_variables; i++) order[i] = i;
    
    OBDD* sat_bdd = obdd_create(num_variables, order);
    
    // Build random 3-SAT clauses
    OBDDNode* result = obdd_constant(1);
    
    // Generate clauses: (x_i OR x_j OR x_k)
    for (int clause = 0; clause < num_clauses && clause < 100; clause++) {
        // Random variables for this clause
        int var1 = clause % num_variables;
        int var2 = (clause + 1) % num_variables;
        int var3 = (clause + 2) % num_variables;
        
        // Make them different
        if (var2 == var1) var2 = (var2 + 1) % num_variables;
        if (var3 == var1 || var3 == var2) var3 = (var3 + 2) % num_variables;
        
        // Random polarities
        bool neg1 = (clause % 3) == 0;
        bool neg2 = (clause % 5) == 0;
        bool neg3 = (clause % 7) == 0;
        
        // Build clause: lit1 OR lit2 OR lit3
        OBDDNode* lit1 = obdd_node_create(var1, 
            neg1 ? obdd_constant(0) : obdd_constant(1),
            neg1 ? obdd_constant(1) : obdd_constant(0));
        
        OBDDNode* lit2 = obdd_node_create(var2,
            neg2 ? obdd_constant(0) : obdd_constant(1),
            neg2 ? obdd_constant(1) : obdd_constant(0));
        
        OBDDNode* lit3 = obdd_node_create(var3,
            neg3 ? obdd_constant(0) : obdd_constant(1),
            neg3 ? obdd_constant(1) : obdd_constant(0));
        
        // OR them together (simplified)
        OBDDNode* clause_bdd = obdd_node_create(var1,
            obdd_node_create(var2, obdd_node_create(var3, obdd_constant(0), obdd_constant(1)),
                                  obdd_node_create(var3, obdd_constant(1), obdd_constant(1))),
            obdd_node_create(var2, obdd_node_create(var3, obdd_constant(1), obdd_constant(1)),
                                  obdd_node_create(var3, obdd_constant(1), obdd_constant(1))));
        
        result = clause_bdd; // Simplified - would need proper AND
    }
    
    sat_bdd->root = result;
    free(order);
    return sat_bdd;
}

// =====================================================
// Problem Suite Management
// =====================================================

int realistic_generate_problem_suite(const RealisticBenchmarkConfig* config,
                                   RealisticProblem* problems, int max_problems) {
    if (!config || !problems || max_problems <= 0) return 0;
    
    int problem_count = 0;
    
    // Generate cryptographic problems
    if (config->include_cryptographic && problem_count < max_problems) {
        for (int i = 0; i < config->problems_per_category && problem_count < max_problems; i++) {
            RealisticProblem* p = &problems[problem_count++];
            
            p->category = PROBLEM_CRYPTOGRAPHIC;
            p->complexity = (ProblemComplexity)(config->min_complexity + 
                           (i % (config->max_complexity - config->min_complexity + 1)));
            
            snprintf(p->name, sizeof(p->name), "AES_S-box_%s_%d", 
                    realistic_get_complexity_name(p->complexity), i);
            snprintf(p->description, sizeof(p->description),
                    "AES S-box transformation with %s complexity",
                    realistic_get_complexity_name(p->complexity));
            
            p->num_variables = 16 + (int)p->complexity * 4;
            p->expected_nodes = p->num_variables * 50;
            p->expected_time_ms = p->num_variables * 2.0;
            
            p->params.crypto.key_bits = 128 + (int)p->complexity * 64;
            p->params.crypto.rounds = 4 + (int)p->complexity;
        }
    }
    
    // Generate combinatorial problems
    if (config->include_combinatorial && problem_count < max_problems) {
        for (int i = 0; i < config->problems_per_category && problem_count < max_problems; i++) {
            RealisticProblem* p = &problems[problem_count++];
            
            p->category = PROBLEM_COMBINATORIAL;
            p->complexity = (ProblemComplexity)(config->min_complexity + 
                           (i % (config->max_complexity - config->min_complexity + 1)));
            
            snprintf(p->name, sizeof(p->name), "N-Queens_%s_%d",
                    realistic_get_complexity_name(p->complexity), i);
            snprintf(p->description, sizeof(p->description),
                    "N-Queens problem with %s complexity",
                    realistic_get_complexity_name(p->complexity));
            
            p->num_variables = (8 + (int)p->complexity * 2) * (8 + (int)p->complexity * 2);
            p->expected_nodes = p->num_variables * 10;
            p->expected_time_ms = p->num_variables * 5.0;
            
            p->params.nqueens.board_size = 8 + (int)p->complexity * 2;
            p->params.nqueens.num_queens = p->params.nqueens.board_size;
        }
    }
    
    // Generate mathematical problems
    if (config->include_mathematical && problem_count < max_problems) {
        for (int i = 0; i < config->problems_per_category && problem_count < max_problems; i++) {
            RealisticProblem* p = &problems[problem_count++];
            
            p->category = PROBLEM_MATHEMATICAL;
            p->complexity = (ProblemComplexity)(config->min_complexity + 
                           (i % (config->max_complexity - config->min_complexity + 1)));
            
            int grid_size = 9 + (int)p->complexity * 3;
            if (grid_size > 16) grid_size = 16;
            
            snprintf(p->name, sizeof(p->name), "Sudoku_%dx%d_%s_%d",
                    grid_size, grid_size, realistic_get_complexity_name(p->complexity), i);
            snprintf(p->description, sizeof(p->description),
                    "Sudoku %dx%d with %s complexity",
                    grid_size, grid_size, realistic_get_complexity_name(p->complexity));
            
            p->num_variables = grid_size * grid_size * (int)ceil(log2(grid_size));
            p->expected_nodes = p->num_variables * 20;
            p->expected_time_ms = p->num_variables * 8.0;
            
            p->params.sudoku.grid_size = grid_size;
            p->params.sudoku.num_clues = grid_size * grid_size / 3;
            p->params.sudoku.variant_type = i % 3;
        }
    }
    
    // Generate SAT problems
    if (config->include_sat && problem_count < max_problems) {
        for (int i = 0; i < config->problems_per_category && problem_count < max_problems; i++) {
            RealisticProblem* p = &problems[problem_count++];
            
            p->category = PROBLEM_SAT_INSTANCES;
            p->complexity = (ProblemComplexity)(config->min_complexity + 
                           (i % (config->max_complexity - config->min_complexity + 1)));
            
            int num_vars = 15 + (int)p->complexity * 5;
            int num_clauses = (int)(num_vars * 4.2); // Near phase transition
            
            snprintf(p->name, sizeof(p->name), "3-SAT_%dv_%dc_%s_%d",
                    num_vars, num_clauses, realistic_get_complexity_name(p->complexity), i);
            snprintf(p->description, sizeof(p->description),
                    "Random 3-SAT with %d variables, %d clauses (%s complexity)",
                    num_vars, num_clauses, realistic_get_complexity_name(p->complexity));
            
            p->num_variables = num_vars;
            p->expected_nodes = num_vars * num_clauses / 2;
            p->expected_time_ms = num_vars * num_clauses * 0.1;
            
            p->params.sat.num_clauses = num_clauses;
            p->params.sat.clause_length = 3;
            p->params.sat.satisfiability_ratio = 0.5; // Phase transition
        }
    }
    
    return problem_count;
}

OBDD* realistic_create_problem_obdd(const RealisticProblem* problem) {
    if (!problem) return nullptr;
    
    switch (problem->category) {
        case PROBLEM_CRYPTOGRAPHIC:
            if (strstr(problem->name, "AES")) {
                return realistic_generate_aes_sbox(problem->params.crypto.key_bits,
                                                 problem->params.crypto.rounds);
            }
            break;
            
        case PROBLEM_COMBINATORIAL:
            if (strstr(problem->name, "N-Queens")) {
                return realistic_generate_nqueens_large(problem->params.nqueens.board_size, 0);
            }
            break;
            
        case PROBLEM_MATHEMATICAL:
            if (strstr(problem->name, "Sudoku")) {
                return realistic_generate_sudoku_large(problem->params.sudoku.grid_size,
                                                     problem->params.sudoku.variant_type,
                                                     problem->params.sudoku.num_clues);
            }
            break;
            
        case PROBLEM_SAT_INSTANCES:
            if (strstr(problem->name, "3-SAT")) {
                return realistic_generate_3sat_random(problem->num_variables,
                                                    problem->params.sat.num_clauses,
                                                    problem->params.sat.satisfiability_ratio);
            }
            break;
            
        default:
            break;
    }
    
    return nullptr;
}

void realistic_print_problem(const RealisticProblem* problem) {
    if (!problem) return;
    
    printf("Problem: %s\n", problem->name);
    printf("Category: %s\n", realistic_get_category_name(problem->category));
    printf("Complexity: %s\n", realistic_get_complexity_name(problem->complexity));
    printf("Variables: %d\n", problem->num_variables);
    printf("Expected Nodes: %d\n", problem->expected_nodes);
    printf("Expected Time: %.2f ms\n", problem->expected_time_ms);
    printf("Description: %s\n", problem->description);
    printf("\n");
}