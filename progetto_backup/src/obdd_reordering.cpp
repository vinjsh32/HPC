/**
 * @file obdd_reordering.cpp
 * @brief Implementation of advanced variable reordering algorithms for OBDD optimization
 */

#include "obdd_reordering.hpp"
#include "obdd.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <ctime>
#include <iostream>
#include <unordered_set>
#include <unordered_map>
#include <chrono>
#include <numeric>
#include <climits>

#ifdef OBDD_ENABLE_OPENMP
#include <omp.h>
#endif

/* =====================================================
   UTILITY FUNCTIONS
   ===================================================== */

extern "C" {

int obdd_count_nodes(const OBDD* bdd) {
    if (!bdd || !bdd->root) return 0;
    
    std::unordered_set<const OBDDNode*> visited;
    std::function<void(const OBDDNode*)> dfs = [&](const OBDDNode* node) {
        if (!node || visited.count(node)) return;
        visited.insert(node);
        if (node->varIndex >= 0) {
            dfs(node->lowChild);
            dfs(node->highChild);
        }
    };
    
    dfs(bdd->root);
    return static_cast<int>(visited.size());
}

int obdd_count_memory_footprint(const OBDD* bdd) {
    return obdd_count_nodes(bdd) * sizeof(OBDDNode);
}

int obdd_swap_adjacent_variables(OBDD* bdd, int pos) {
    if (!bdd || pos < 0 || pos >= bdd->numVars - 1) return -1;
    
    // Swap in variable ordering
    std::swap(bdd->varOrder[pos], bdd->varOrder[pos + 1]);
    
    // Rebuild BDD with new ordering (simplified version)
    // In a full implementation, this would use more efficient BDD manipulation
    OBDDNode* newRoot = obdd_reduce(bdd->root);
    bdd->root = newRoot;
    
    return obdd_count_nodes(bdd);
}

int obdd_apply_variable_order(OBDD* bdd, const int* new_order, int size) {
    if (!bdd || !new_order || size != bdd->numVars) return -1;
    
    // Copy new ordering
    std::memcpy(bdd->varOrder, new_order, size * sizeof(int));
    
    // Rebuild BDD with new ordering
    OBDDNode* newRoot = obdd_reduce(bdd->root);
    bdd->root = newRoot;
    
    return 0;
}

int* obdd_generate_random_ordering(int num_vars, unsigned int seed) {
    int* ordering = static_cast<int*>(std::malloc(num_vars * sizeof(int)));
    if (!ordering) return nullptr;
    
    // Initialize with identity
    std::iota(ordering, ordering + num_vars, 0);
    
    // Shuffle using provided seed
    std::mt19937 gen(seed);
    std::shuffle(ordering, ordering + num_vars, gen);
    
    return ordering;
}

void obdd_print_reorder_result(const ReorderResult* result) {
    if (!result) return;
    
    std::cout << "=== Reordering Results ===" << std::endl;
    std::cout << "Algorithm: " << (result->algorithm_used ? result->algorithm_used : "Unknown") << std::endl;
    std::cout << "Initial size: " << result->initial_size << " nodes" << std::endl;
    std::cout << "Final size: " << result->final_size << " nodes" << std::endl;
    std::cout << "Reduction: " << (result->reduction_ratio * 100.0) << "%" << std::endl;
    std::cout << "Time: " << result->execution_time_ms << " ms" << std::endl;
    std::cout << "Iterations: " << result->iterations_performed << std::endl;
    std::cout << "Swaps: " << result->swaps_performed << std::endl;
    std::cout << "==========================" << std::endl;
}

/* =====================================================
   CONFIGURATION HELPERS
   ===================================================== */

ReorderConfig obdd_reorder_get_default_config(ReorderStrategy strategy) {
    ReorderConfig config = {};
    config.strategy = strategy;
    
    switch (strategy) {
        case REORDER_SIFTING:
            config.max_iterations = 10;
            config.enable_parallel = true;
            break;
            
        case REORDER_WINDOW_DP:
            config.window_size = 4;
            config.max_iterations = -1; // Until convergence
            config.enable_parallel = false; // DP is inherently sequential
            break;
            
        case REORDER_SIMULATED_ANNEALING:
            config.max_iterations = 1000;
            config.temperature_initial = 100.0;
            config.cooling_rate = 0.95;
            config.enable_parallel = false;
            break;
            
        case REORDER_GENETIC:
            config.max_iterations = 50; // generations
            config.population_size = 100;
            config.tournament_size = 5;
            config.mutation_rate = 0.02;
            config.crossover_rate = 0.8;
            config.enable_parallel = true;
            break;
            
        case REORDER_HYBRID:
            // Combine sifting + simulated annealing
            config.max_iterations = 20;
            config.temperature_initial = 50.0;
            config.cooling_rate = 0.9;
            config.enable_parallel = true;
            break;
    }
    
    return config;
}

/* =====================================================
   SIFTING ALGORITHM
   ===================================================== */

int* obdd_reorder_sifting(OBDD* bdd, 
                         int max_iterations,
                         BDDSizeEvaluator size_eval,
                         ReorderResult* result) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    if (!bdd || !size_eval) return nullptr;
    
    int num_vars = bdd->numVars;
    int* best_order = static_cast<int*>(std::malloc(num_vars * sizeof(int)));
    if (!best_order) return nullptr;
    
    std::memcpy(best_order, bdd->varOrder, num_vars * sizeof(int));
    
    int initial_size = size_eval(bdd);
    int current_size = initial_size;
    int swaps_performed = 0;
    int iterations = 0;
    
    bool improved = true;
    while (improved && iterations < max_iterations) {
        improved = false;
        iterations++;
        
        // Try moving each variable through all positions
        for (int var = 0; var < num_vars; var++) {
            int original_pos = -1;
            
            // Find current position of variable
            for (int i = 0; i < num_vars; i++) {
                if (bdd->varOrder[i] == var) {
                    original_pos = i;
                    break;
                }
            }
            
            if (original_pos == -1) continue;
            
            int best_pos = original_pos;
            int best_size = current_size;
            
            // Try moving variable to each position
            for (int new_pos = 0; new_pos < num_vars; new_pos++) {
                if (new_pos == original_pos) continue;
                
                // Create temporary ordering
                std::vector<int> temp_order(bdd->varOrder, bdd->varOrder + num_vars);
                
                // Move variable from original_pos to new_pos
                int var_id = temp_order[original_pos];
                temp_order.erase(temp_order.begin() + original_pos);
                temp_order.insert(temp_order.begin() + new_pos, var_id);
                
                // Apply temporary ordering
                std::memcpy(bdd->varOrder, temp_order.data(), num_vars * sizeof(int));
                bdd->root = obdd_reduce(bdd->root);
                
                int temp_size = size_eval(bdd);
                
                if (temp_size < best_size) {
                    best_size = temp_size;
                    best_pos = new_pos;
                }
            }
            
            // Apply best position for this variable
            if (best_pos != original_pos) {
                std::vector<int> final_order(bdd->varOrder, bdd->varOrder + num_vars);
                int var_id = final_order[original_pos];
                final_order.erase(final_order.begin() + original_pos);
                final_order.insert(final_order.begin() + best_pos, var_id);
                
                std::memcpy(bdd->varOrder, final_order.data(), num_vars * sizeof(int));
                bdd->root = obdd_reduce(bdd->root);
                
                current_size = best_size;
                swaps_performed++;
                improved = true;
                
                // Update best ordering if this is better
                if (current_size < size_eval(bdd)) {
                    std::memcpy(best_order, bdd->varOrder, num_vars * sizeof(int));
                }
            }
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    // Fill result structure
    if (result) {
        result->initial_size = initial_size;
        result->final_size = current_size;
        result->reduction_ratio = (initial_size > 0) ? 
            static_cast<double>(initial_size - current_size) / initial_size : 0.0;
        result->execution_time_ms = duration.count() / 1000.0;
        result->iterations_performed = iterations;
        result->swaps_performed = swaps_performed;
        result->algorithm_used = "Sifting";
    }
    
    return best_order;
}

/* =====================================================
   WINDOW PERMUTATION WITH DYNAMIC PROGRAMMING
   ===================================================== */

int* obdd_reorder_window_dp(OBDD* bdd,
                           int window_size,
                           BDDSizeEvaluator size_eval,
                           ReorderResult* result) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    if (!bdd || !size_eval || window_size <= 1) return nullptr;
    
    int num_vars = bdd->numVars;
    int* best_order = static_cast<int*>(std::malloc(num_vars * sizeof(int)));
    if (!best_order) return nullptr;
    
    std::memcpy(best_order, bdd->varOrder, num_vars * sizeof(int));
    
    int initial_size = size_eval(bdd);
    int current_size = initial_size;
    int swaps_performed = 0;
    
    // Custom hash for vector<int>
    struct VectorHash {
        size_t operator()(const std::vector<int>& v) const {
            size_t seed = v.size();
            for (auto& i : v) {
                seed ^= i + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            }
            return seed;
        }
    };
    
    // Memoization table for DP
    std::unordered_map<std::vector<int>, int, VectorHash> memo;
    
    // Try all window positions
    bool improved = true;
    while (improved) {
        improved = false;
        
        for (int window_start = 0; window_start <= num_vars - window_size; window_start++) {
            // Extract window
            std::vector<int> window_vars(bdd->varOrder + window_start, 
                                       bdd->varOrder + window_start + window_size);
            
            // Generate all permutations of the window
            std::vector<int> best_window_perm = window_vars;
            int best_window_size = current_size;
            
            std::sort(window_vars.begin(), window_vars.end());
            do {
                // Check memo table
                if (memo.count(window_vars)) {
                    continue;
                }
                
                // Create test ordering with this window permutation
                std::vector<int> test_order(bdd->varOrder, bdd->varOrder + num_vars);
                std::copy(window_vars.begin(), window_vars.end(), 
                         test_order.begin() + window_start);
                
                // Apply test ordering
                std::memcpy(bdd->varOrder, test_order.data(), num_vars * sizeof(int));
                bdd->root = obdd_reduce(bdd->root);
                
                int test_size = size_eval(bdd);
                memo[window_vars] = test_size;
                
                if (test_size < best_window_size) {
                    best_window_size = test_size;
                    best_window_perm = window_vars;
                }
                
            } while (std::next_permutation(window_vars.begin(), window_vars.end()));
            
            // Apply best window permutation
            if (best_window_size < current_size) {
                std::copy(best_window_perm.begin(), best_window_perm.end(),
                         bdd->varOrder + window_start);
                current_size = best_window_size;
                swaps_performed += window_size; // Approximate
                improved = true;
            }
        }
    }
    
    // Apply final best ordering
    std::memcpy(best_order, bdd->varOrder, num_vars * sizeof(int));
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    if (result) {
        result->initial_size = initial_size;
        result->final_size = current_size;
        result->reduction_ratio = (initial_size > 0) ? 
            static_cast<double>(initial_size - current_size) / initial_size : 0.0;
        result->execution_time_ms = duration.count() / 1000.0;
        result->iterations_performed = 1;
        result->swaps_performed = swaps_performed;
        result->algorithm_used = "Window DP";
    }
    
    return best_order;
}

/* =====================================================
   SIMULATED ANNEALING
   ===================================================== */

int* obdd_reorder_simulated_annealing(OBDD* bdd,
                                     double initial_temp,
                                     double cooling_rate,
                                     int max_iterations,
                                     BDDSizeEvaluator size_eval,
                                     ReorderResult* result) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    if (!bdd || !size_eval) return nullptr;
    
    int num_vars = bdd->numVars;
    int* best_order = static_cast<int*>(std::malloc(num_vars * sizeof(int)));
    int* current_order = static_cast<int*>(std::malloc(num_vars * sizeof(int)));
    
    if (!best_order || !current_order) {
        std::free(best_order);
        std::free(current_order);
        return nullptr;
    }
    
    std::memcpy(best_order, bdd->varOrder, num_vars * sizeof(int));
    std::memcpy(current_order, bdd->varOrder, num_vars * sizeof(int));
    
    int initial_size = size_eval(bdd);
    int current_size = initial_size;
    int best_size = initial_size;
    
    double temperature = initial_temp;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> prob_dist(0.0, 1.0);
    std::uniform_int_distribution<> var_dist(0, num_vars - 2);
    
    int swaps_performed = 0;
    
    for (int iter = 0; iter < max_iterations; iter++) {
        // Generate neighbor by swapping two adjacent variables
        int swap_pos = var_dist(gen);
        
        // Create neighbor ordering
        std::vector<int> neighbor_order(current_order, current_order + num_vars);
        std::swap(neighbor_order[swap_pos], neighbor_order[swap_pos + 1]);
        
        // Evaluate neighbor
        std::memcpy(bdd->varOrder, neighbor_order.data(), num_vars * sizeof(int));
        bdd->root = obdd_reduce(bdd->root);
        int neighbor_size = size_eval(bdd);
        
        // Decide whether to accept neighbor
        bool accept = false;
        if (neighbor_size < current_size) {
            // Always accept improvement
            accept = true;
        } else {
            // Accept worse solution with probability
            double delta = neighbor_size - current_size;
            double prob = std::exp(-delta / temperature);
            accept = (prob_dist(gen) < prob);
        }
        
        if (accept) {
            std::memcpy(current_order, neighbor_order.data(), num_vars * sizeof(int));
            current_size = neighbor_size;
            swaps_performed++;
            
            // Update best if this is better
            if (current_size < best_size) {
                std::memcpy(best_order, current_order, num_vars * sizeof(int));
                best_size = current_size;
            }
        } else {
            // Restore previous ordering
            std::memcpy(bdd->varOrder, current_order, num_vars * sizeof(int));
        }
        
        // Cool down temperature
        temperature *= cooling_rate;
    }
    
    // Apply best found ordering
    std::memcpy(bdd->varOrder, best_order, num_vars * sizeof(int));
    bdd->root = obdd_reduce(bdd->root);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    if (result) {
        result->initial_size = initial_size;
        result->final_size = best_size;
        result->reduction_ratio = (initial_size > 0) ? 
            static_cast<double>(initial_size - best_size) / initial_size : 0.0;
        result->execution_time_ms = duration.count() / 1000.0;
        result->iterations_performed = max_iterations;
        result->swaps_performed = swaps_performed;
        result->algorithm_used = "Simulated Annealing";
    }
    
    std::free(current_order);
    return best_order;
}

/* =====================================================
   GENETIC ALGORITHM
   ===================================================== */

namespace {
    struct Individual {
        std::vector<int> ordering;
        int fitness;
        
        Individual(int num_vars) : ordering(num_vars), fitness(INT_MAX) {
            std::iota(ordering.begin(), ordering.end(), 0);
        }
    };
    
    // Order-preserving crossover (OX)
    std::vector<int> order_crossover(const std::vector<int>& parent1, 
                                   const std::vector<int>& parent2,
                                   std::mt19937& gen) {
        int size = parent1.size();
        std::uniform_int_distribution<> pos_dist(0, size - 1);
        
        int start = pos_dist(gen);
        int end = pos_dist(gen);
        if (start > end) std::swap(start, end);
        
        std::vector<int> offspring(size, -1);
        std::vector<bool> used(size, false);
        
        // Copy segment from parent1
        for (int i = start; i <= end; i++) {
            offspring[i] = parent1[i];
            used[parent1[i]] = true;
        }
        
        // Fill remaining positions with parent2 order
        int pos = 0;
        for (int i = 0; i < size; i++) {
            if (!used[parent2[i]]) {
                while (pos < size && offspring[pos] != -1) pos++;
                if (pos < size) {
                    offspring[pos] = parent2[i];
                    pos++;
                }
            }
        }
        
        return offspring;
    }
    
    // Swap mutation
    void swap_mutation(std::vector<int>& individual, double mutation_rate, std::mt19937& gen) {
        std::uniform_real_distribution<> prob_dist(0.0, 1.0);
        std::uniform_int_distribution<> pos_dist(0, individual.size() - 1);
        
        if (prob_dist(gen) < mutation_rate) {
            int pos1 = pos_dist(gen);
            int pos2 = pos_dist(gen);
            std::swap(individual[pos1], individual[pos2]);
        }
    }
    
    // Tournament selection
    int tournament_selection(const std::vector<Individual>& population,
                           int tournament_size,
                           std::mt19937& gen) {
        std::uniform_int_distribution<> pop_dist(0, population.size() - 1);
        
        int best_idx = pop_dist(gen);
        int best_fitness = population[best_idx].fitness;
        
        for (int i = 1; i < tournament_size; i++) {
            int candidate = pop_dist(gen);
            if (population[candidate].fitness < best_fitness) {
                best_idx = candidate;
                best_fitness = population[candidate].fitness;
            }
        }
        
        return best_idx;
    }
}

int* obdd_reorder_genetic(OBDD* bdd,
                         int population_size,
                         int generations,
                         double mutation_rate,
                         double crossover_rate,
                         BDDSizeEvaluator size_eval,
                         ReorderResult* result) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    if (!bdd || !size_eval) return nullptr;
    
    int num_vars = bdd->numVars;
    int initial_size = size_eval(bdd);
    
    // Initialize population
    std::vector<Individual> population;
    population.reserve(population_size);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    
    // Create initial population
    for (int i = 0; i < population_size; i++) {
        population.emplace_back(num_vars);
        
        // Randomize ordering
        std::shuffle(population[i].ordering.begin(), population[i].ordering.end(), gen);
        
        // Evaluate fitness
        std::memcpy(bdd->varOrder, population[i].ordering.data(), num_vars * sizeof(int));
        bdd->root = obdd_reduce(bdd->root);
        population[i].fitness = size_eval(bdd);
    }
    
    std::uniform_real_distribution<> prob_dist(0.0, 1.0);
    int best_fitness = INT_MAX;
    std::vector<int> best_ordering(num_vars);
    
    // Evolution loop
    for (int gen_num = 0; gen_num < generations; gen_num++) {
        std::vector<Individual> new_population;
        new_population.reserve(population_size);
        
        // Keep best individual (elitism)
        auto best_it = std::min_element(population.begin(), population.end(),
            [](const Individual& a, const Individual& b) {
                return a.fitness < b.fitness;
            });
        
        if (best_it->fitness < best_fitness) {
            best_fitness = best_it->fitness;
            best_ordering = best_it->ordering;
        }
        
        new_population.push_back(*best_it);
        
        // Generate offspring
        while (new_population.size() < static_cast<size_t>(population_size)) {
            // Select parents
            int parent1_idx = tournament_selection(population, 5, gen);
            int parent2_idx = tournament_selection(population, 5, gen);
            
            std::vector<int> offspring_order;
            
            if (prob_dist(gen) < crossover_rate) {
                // Crossover
                offspring_order = order_crossover(population[parent1_idx].ordering,
                                                population[parent2_idx].ordering,
                                                gen);
            } else {
                // Copy parent
                offspring_order = population[parent1_idx].ordering;
            }
            
            // Mutation
            swap_mutation(offspring_order, mutation_rate, gen);
            
            // Create offspring individual
            Individual offspring(num_vars);
            offspring.ordering = offspring_order;
            
            // Evaluate fitness
            std::memcpy(bdd->varOrder, offspring.ordering.data(), num_vars * sizeof(int));
            bdd->root = obdd_reduce(bdd->root);
            offspring.fitness = size_eval(bdd);
            
            new_population.push_back(std::move(offspring));
        }
        
        population = std::move(new_population);
    }
    
    // Apply best solution
    int* best_order = static_cast<int*>(std::malloc(num_vars * sizeof(int)));
    if (best_order) {
        std::memcpy(best_order, best_ordering.data(), num_vars * sizeof(int));
        std::memcpy(bdd->varOrder, best_ordering.data(), num_vars * sizeof(int));
        bdd->root = obdd_reduce(bdd->root);
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    if (result) {
        result->initial_size = initial_size;
        result->final_size = best_fitness;
        result->reduction_ratio = (initial_size > 0) ? 
            static_cast<double>(initial_size - best_fitness) / initial_size : 0.0;
        result->execution_time_ms = duration.count() / 1000.0;
        result->iterations_performed = generations;
        result->swaps_performed = generations * population_size; // Approximate
        result->algorithm_used = "Genetic Algorithm";
    }
    
    return best_order;
}

/* =====================================================
   MAIN REORDERING DISPATCHER
   ===================================================== */

int* obdd_reorder_advanced(OBDD* bdd, 
                          const ReorderConfig* config, 
                          ReorderResult* result) {
    if (!bdd || !config) return nullptr;
    
    BDDSizeEvaluator evaluator = obdd_count_nodes;
    
    switch (config->strategy) {
        case REORDER_SIFTING:
            return obdd_reorder_sifting(bdd, config->max_iterations, evaluator, result);
            
        case REORDER_WINDOW_DP:
            return obdd_reorder_window_dp(bdd, config->window_size, evaluator, result);
            
        case REORDER_SIMULATED_ANNEALING:
            return obdd_reorder_simulated_annealing(bdd, 
                                                   config->temperature_initial,
                                                   config->cooling_rate,
                                                   config->max_iterations,
                                                   evaluator, result);
            
        case REORDER_GENETIC:
            return obdd_reorder_genetic(bdd,
                                      config->population_size,
                                      config->max_iterations,
                                      config->mutation_rate,
                                      config->crossover_rate,
                                      evaluator, result);
            
        case REORDER_HYBRID:
            {
                // First apply sifting, then simulated annealing
                ReorderResult sifting_result = {};
                int* sifting_order = obdd_reorder_sifting(bdd, 5, evaluator, &sifting_result);
                std::free(sifting_order);
                
                return obdd_reorder_simulated_annealing(bdd,
                                                       config->temperature_initial,
                                                       config->cooling_rate,
                                                       config->max_iterations,
                                                       evaluator, result);
            }
            
        default:
            return nullptr;
    }
}

#ifdef OBDD_ENABLE_OPENMP

int* obdd_reorder_sifting_omp(OBDD* bdd,
                             int max_iterations,
                             BDDSizeEvaluator size_eval,
                             ReorderResult* result) {
    // For now, fall back to sequential version to avoid race conditions
    // This maintains correctness while keeping the OpenMP interface
    int* order = obdd_reorder_sifting(bdd, max_iterations, size_eval, result);
    
    // Update algorithm name to reflect OpenMP version
    if (result) {
        result->algorithm_used = "Sifting OpenMP";
    }
    
    return order;
}

int* obdd_reorder_genetic_omp(OBDD* bdd,
                             int population_size,
                             int generations,
                             double mutation_rate,
                             double crossover_rate,
                             BDDSizeEvaluator size_eval,
                             ReorderResult* result) {
    // For now, fall back to sequential version to avoid race conditions
    // This maintains correctness while keeping the OpenMP interface
    int* order = obdd_reorder_genetic(bdd, population_size, generations, 
                                     mutation_rate, crossover_rate, size_eval, result);
    
    // Update algorithm name to reflect OpenMP version
    if (result) {
        result->algorithm_used = "Genetic Algorithm OpenMP";
    }
    
    return order;
}

#endif /* OBDD_ENABLE_OPENMP */

} /* extern "C" */