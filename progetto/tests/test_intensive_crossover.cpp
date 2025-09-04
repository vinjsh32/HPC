/**
 * @file test_intensive_crossover.cpp
 * @brief Intensive computational test to find real crossover points
 */

#include <gtest/gtest.h>
#include "core/obdd.hpp"
#include <chrono>
#include <iostream>
#include <vector>
#include <iomanip>
#include <omp.h>
#include <fstream>

class IntensiveCrossover : public ::testing::Test {
protected:
    struct Result {
        int variables;
        int iterations;
        long sequential_ms;
        long openmp_ms;
        double speedup;
        bool found_benefit;
    };
    
    std::vector<Result> results;
    
    void SetUp() override {
        omp_set_num_threads(std::min(8, omp_get_max_threads()));
    }
    
    /**
     * Create computationally intensive BDD operations
     * Multiple iterations of complex operations to accumulate measurable time
     */
    long benchmark_intensive_operations(OBDD* bdd1, OBDD* bdd2, int iterations, bool use_openmp) {
        auto start = std::chrono::high_resolution_clock::now();
        
        std::vector<OBDDNode*> temp_results;
        temp_results.reserve(iterations * 3);
        
        for (int iter = 0; iter < iterations; ++iter) {
            OBDDNode* and_result, *or_result, *xor_result, *composite1, *composite2;
            
            if (use_openmp) {
                // Multiple parallel operations to stress the system
                and_result = obdd_parallel_apply_omp(bdd1, bdd2, OBDD_AND);
                or_result = obdd_parallel_apply_omp(bdd1, bdd2, OBDD_OR);
                xor_result = obdd_parallel_apply_omp(bdd1, bdd2, OBDD_XOR);
                
                // Create temporary BDDs for composite operations
                OBDD* temp1 = obdd_create(bdd1->numVars, bdd1->varOrder);
                OBDD* temp2 = obdd_create(bdd1->numVars, bdd1->varOrder);
                temp1->root = and_result;
                temp2->root = or_result;
                
                composite1 = obdd_parallel_apply_omp(temp1, temp2, OBDD_XOR);
                temp1->root = xor_result;
                temp2->root = composite1;
                composite2 = obdd_parallel_apply_omp(temp1, temp2, OBDD_AND);
                
                obdd_destroy(temp1);
                obdd_destroy(temp2);
            } else {
                // Sequential operations
                and_result = obdd_apply(bdd1, bdd2, OBDD_AND);
                or_result = obdd_apply(bdd1, bdd2, OBDD_OR);
                xor_result = obdd_apply(bdd1, bdd2, OBDD_XOR);
                
                OBDD* temp1 = obdd_create(bdd1->numVars, bdd1->varOrder);
                OBDD* temp2 = obdd_create(bdd1->numVars, bdd1->varOrder);
                temp1->root = and_result;
                temp2->root = or_result;
                
                composite1 = obdd_apply(temp1, temp2, OBDD_XOR);
                temp1->root = xor_result;
                temp2->root = composite1;
                composite2 = obdd_apply(temp1, temp2, OBDD_AND);
                
                obdd_destroy(temp1);
                obdd_destroy(temp2);
            }
            
            // Store results to prevent optimization
            temp_results.push_back(and_result);
            temp_results.push_back(or_result);
            temp_results.push_back(xor_result);
            temp_results.push_back(composite1);
            temp_results.push_back(composite2);
            
            // Periodic cleanup to prevent memory explosion
            if (iter % 10 == 9) {
                temp_results.clear();
            }
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    }
    
    /**
     * Create BDD with exponentially complex structure
     */
    OBDD* create_exponential_bdd(int variables) {
        std::vector<int> order(variables);
        for (int i = 0; i < variables; ++i) order[i] = i;
        
        OBDD* bdd = obdd_create(variables, order.data());
        
        // Create a structure that forces many recursive apply calls
        OBDDNode* current = obdd_constant(0);
        
        for (int var = variables - 1; var >= 0; --var) {
            // Create complex pattern that doesn't reduce easily
            OBDDNode* pattern1, *pattern2;
            
            if (var % 4 == 0) {
                pattern1 = obdd_node_create(var, current, obdd_constant(1));
                pattern2 = obdd_node_create(var, obdd_constant(1), current);
            } else if (var % 4 == 1) {
                pattern1 = obdd_node_create(var, obdd_constant(0), current);
                pattern2 = obdd_node_create(var, current, obdd_constant(0));
            } else if (var % 4 == 2) {
                pattern1 = obdd_node_create(var, current, current);
                pattern2 = obdd_node_create(var, obdd_constant(1), obdd_constant(0));
            } else {
                pattern1 = obdd_node_create(var, obdd_constant(0), obdd_constant(1));
                pattern2 = obdd_node_create(var, current, obdd_constant(1));
            }
            
            current = (var % 2 == 0) ? pattern1 : pattern2;
        }
        
        bdd->root = current;
        return bdd;
    }
    
    void save_intensive_results() {
        std::ofstream file("intensive_crossover_results.csv");
        file << "Variables,Iterations,Sequential_ms,OpenMP_ms,Speedup,Found_Benefit\n";
        
        for (const auto& result : results) {
            file << result.variables << ","
                 << result.iterations << ","
                 << result.sequential_ms << ","
                 << result.openmp_ms << ","
                 << std::fixed << std::setprecision(3) << result.speedup << ","
                 << (result.found_benefit ? "YES" : "NO") << "\n";
        }
        file.close();
        std::cout << "\n📊 Intensive results saved to: intensive_crossover_results.csv" << std::endl;
    }
};

TEST_F(IntensiveCrossover, FindRealCrossover) {
    std::cout << "\n=== INTENSIVE CROSSOVER ANALYSIS ===" << std::endl;
    std::cout << "Using high iteration counts to create measurable computation" << std::endl;
    
    std::cout << std::setw(8) << "Vars"
              << std::setw(10) << "Iters"
              << std::setw(12) << "Sequential"
              << std::setw(12) << "OpenMP"
              << std::setw(12) << "Speedup"
              << std::setw(10) << "Status" << std::endl;
    std::cout << std::string(64, '-') << std::endl;
    
    bool found_crossover = false;
    
    // Test with increasing complexity
    for (int vars = 12; vars <= 24; vars += 2) {
        Result result;
        result.variables = vars;
        
        // Scale iterations based on variable count to maintain reasonable timing
        result.iterations = std::max(50, 300 - vars * 8);
        
        std::cout << "Testing " << vars << " variables with " << result.iterations << " iterations... ";
        std::cout.flush();
        
        // Create complex BDDs
        OBDD* bdd1 = create_exponential_bdd(vars);
        OBDD* bdd2 = create_exponential_bdd(vars);
        
        // Sequential benchmark
        std::cout << "seq... ";
        std::cout.flush();
        result.sequential_ms = benchmark_intensive_operations(bdd1, bdd2, result.iterations, false);
        
        if (result.sequential_ms <= 0) {
            std::cout << "❌ Sequential failed" << std::endl;
            obdd_destroy(bdd1);
            obdd_destroy(bdd2);
            continue;
        }
        
        // OpenMP benchmark
        std::cout << "omp... ";
        std::cout.flush();
        result.openmp_ms = benchmark_intensive_operations(bdd1, bdd2, result.iterations, true);
        
        if (result.openmp_ms <= 0) {
            std::cout << "❌ OpenMP failed" << std::endl;
            obdd_destroy(bdd1);
            obdd_destroy(bdd2);
            continue;
        }
        
        // Calculate speedup
        result.speedup = (double)result.sequential_ms / result.openmp_ms;
        result.found_benefit = result.speedup > 1.0;
        
        // Print results
        std::cout << std::setw(8) << vars
                  << std::setw(10) << result.iterations
                  << std::setw(12) << result.sequential_ms << "ms"
                  << std::setw(12) << result.openmp_ms << "ms"
                  << std::setw(12) << std::fixed << std::setprecision(2) << result.speedup << "x";
        
        if (result.found_benefit) {
            std::cout << std::setw(10) << "🎯 BENEFIT!";
            if (!found_crossover) {
                std::cout << " ← FIRST CROSSOVER!";
                found_crossover = true;
            }
        } else {
            std::cout << std::setw(10) << "No benefit";
        }
        std::cout << std::endl;
        
        results.push_back(result);
        
        // Cleanup
        obdd_destroy(bdd1);
        obdd_destroy(bdd2);
        
        // Safety check - stop if taking too long
        if (result.sequential_ms > 15000) {  // 15 seconds
            std::cout << "\n⚠️  Tests taking too long, stopping for safety" << std::endl;
            break;
        }
        
        // If we found crossover and tested enough, we can conclude
        if (found_crossover && vars >= 20) {
            std::cout << "\n✅ Crossover found and validated, stopping analysis" << std::endl;
            break;
        }
    }
    
    // Analysis summary
    std::cout << "\n=== INTENSIVE CROSSOVER SUMMARY ===" << std::endl;
    
    if (found_crossover) {
        auto first_benefit = std::find_if(results.begin(), results.end(),
            [](const Result& r) { return r.found_benefit; });
        
        if (first_benefit != results.end()) {
            std::cout << "🎯 FIRST OPENMP BENEFIT at " << first_benefit->variables 
                      << " variables with " << first_benefit->iterations << " iterations" << std::endl;
            std::cout << "   Speedup: " << first_benefit->speedup << "x" << std::endl;
            std::cout << "   Sequential time: " << first_benefit->sequential_ms << "ms" << std::endl;
            std::cout << "   OpenMP time: " << first_benefit->openmp_ms << "ms" << std::endl;
        }
        
        // Find best speedup
        auto best_speedup = *std::max_element(results.begin(), results.end(),
            [](const Result& a, const Result& b) { return a.speedup < b.speedup; });
        
        std::cout << "🏆 BEST SPEEDUP: " << best_speedup.speedup << "x at " 
                  << best_speedup.variables << " variables" << std::endl;
        
    } else {
        std::cout << "❌ No OpenMP benefit found in tested range" << std::endl;
        
        if (!results.empty()) {
            auto best_attempt = *std::max_element(results.begin(), results.end(),
                [](const Result& a, const Result& b) { return a.speedup < b.speedup; });
            
            std::cout << "   Best attempt: " << best_attempt.speedup << "x at " 
                      << best_attempt.variables << " variables" << std::endl;
            std::cout << "   (Sequential: " << best_attempt.sequential_ms 
                      << "ms, OpenMP: " << best_attempt.openmp_ms << "ms)" << std::endl;
        }
    }
    
    save_intensive_results();
    
    // Validate we collected useful data
    ASSERT_FALSE(results.empty()) << "No benchmark data collected";
    
    // Check if we found meaningful computation times
    bool found_measurable_time = std::any_of(results.begin(), results.end(),
        [](const Result& r) { return r.sequential_ms >= 10; }); // At least 10ms
    
    if (found_measurable_time) {
        std::cout << "\n✅ Found measurable computation times for analysis" << std::endl;
    } else {
        std::cout << "\n⚠️  All computation times very small - may need larger problems" << std::endl;
    }
}