/**
 * @file test_high_volume_crossover.cpp
 * @brief High-volume iteration test to find crossover through computational load
 */

#include <gtest/gtest.h>
#include "core/obdd.hpp"
#include <chrono>
#include <iostream>
#include <vector>
#include <iomanip>
#include <omp.h>
#include <fstream>

class HighVolumeCrossover : public ::testing::Test {
protected:
    struct VolumeResult {
        int variables;
        int iterations;
        long sequential_ms;
        long openmp_ms;
        double speedup;
        double efficiency;
        bool crossover_found;
    };
    
    std::vector<VolumeResult> results;
    
    void SetUp() override {
        int max_threads = omp_get_max_threads();
        omp_set_num_threads(std::min(8, max_threads));
        std::cout << "\n🔥 HIGH-VOLUME CROSSOVER ANALYSIS" << std::endl;
        std::cout << "Using " << omp_get_max_threads() << " threads" << std::endl;
    }
    
    /**
     * High-volume benchmark with thousands of operations
     */
    long run_volume_benchmark(OBDD* bdd1, OBDD* bdd2, int iterations, bool use_openmp) {
        auto start = std::chrono::high_resolution_clock::now();
        
        // Use volatile to prevent compiler optimization
        volatile int result_count = 0;
        
        for (int iter = 0; iter < iterations; ++iter) {
            OBDDNode* result1, *result2, *result3, *result4;
            
            if (use_openmp) {
                result1 = obdd_parallel_apply_omp(bdd1, bdd2, OBDD_AND);
                result2 = obdd_parallel_apply_omp(bdd1, bdd2, OBDD_OR);
                result3 = obdd_parallel_apply_omp(bdd1, bdd2, OBDD_XOR);
                
                // Create temporary for chaining
                OBDD* temp = obdd_create(bdd1->numVars, bdd1->varOrder);
                temp->root = result1;
                result4 = obdd_parallel_apply_omp(temp, bdd2, OBDD_AND);
                obdd_destroy(temp);
            } else {
                result1 = obdd_apply(bdd1, bdd2, OBDD_AND);
                result2 = obdd_apply(bdd1, bdd2, OBDD_OR);
                result3 = obdd_apply(bdd1, bdd2, OBDD_XOR);
                
                OBDD* temp = obdd_create(bdd1->numVars, bdd1->varOrder);
                temp->root = result1;
                result4 = obdd_apply(temp, bdd2, OBDD_AND);
                obdd_destroy(temp);
            }
            
            // Count results to prevent optimization
            if (result1) result_count++;
            if (result2) result_count++;
            if (result3) result_count++;
            if (result4) result_count++;
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        
        // Use result_count to prevent optimization
        if (result_count < iterations) {
            std::cout << "⚠️  Some operations failed (" << result_count << "/" << (iterations*4) << ")" << std::endl;
        }
        
        return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    }
    
    /**
     * Create fixed-size BDD optimized for repeated operations
     */
    OBDD* create_compute_intensive_bdd(int variables = 14) {
        std::vector<int> order(variables);
        for (int i = 0; i < variables; ++i) order[i] = i;
        
        OBDD* bdd = obdd_create(variables, order.data());
        
        // Create a moderately complex but stable structure
        OBDDNode* current = obdd_constant(0);
        
        for (int var = variables - 1; var >= 0; --var) {
            OBDDNode* low, *high;
            
            switch (var % 3) {
                case 0:
                    low = current;
                    high = (var > variables/2) ? obdd_constant(1) : current;
                    break;
                case 1:
                    low = (var % 2 == 0) ? obdd_constant(0) : current;
                    high = obdd_constant(1);
                    break;
                default:
                    low = obdd_constant(0);
                    high = (var < variables/2) ? current : obdd_constant(1);
                    break;
            }
            
            current = obdd_node_create(var, low, high);
        }
        
        bdd->root = current;
        return bdd;
    }
    
    void save_volume_results() {
        std::ofstream file("high_volume_crossover_results.csv");
        file << "Variables,Iterations,Sequential_ms,OpenMP_ms,Speedup,Efficiency,Crossover_Found\n";
        
        for (const auto& result : results) {
            file << result.variables << ","
                 << result.iterations << ","
                 << result.sequential_ms << ","
                 << result.openmp_ms << ","
                 << std::fixed << std::setprecision(3) << result.speedup << ","
                 << result.efficiency << ","
                 << (result.crossover_found ? "YES" : "NO") << "\n";
        }
        file.close();
        std::cout << "\n📊 High-volume results saved to: high_volume_crossover_results.csv" << std::endl;
    }
};

TEST_F(HighVolumeCrossover, FindVolumeBasedCrossover) {
    std::cout << "\n=== HIGH-VOLUME ITERATION CROSSOVER TEST ===" << std::endl;
    std::cout << "Strategy: Use many iterations on fixed-size problems" << std::endl;
    
    std::cout << std::setw(8) << "Vars"
              << std::setw(10) << "Iters"
              << std::setw(12) << "Sequential"
              << std::setw(12) << "OpenMP"
              << std::setw(10) << "Speedup"
              << std::setw(12) << "Efficiency"
              << std::setw(10) << "Status" << std::endl;
    std::cout << std::string(78, '-') << std::endl;
    
    // Use fixed problem size but increase iteration count
    const int variables = 14;  // Sweet spot size
    
    // Create BDDs once and reuse
    OBDD* bdd1 = create_compute_intensive_bdd(variables);
    OBDD* bdd2 = create_compute_intensive_bdd(variables);
    
    std::vector<int> iteration_counts = {100, 300, 500, 1000, 2000, 3000, 5000, 8000, 10000};
    
    bool found_crossover = false;
    
    for (int iterations : iteration_counts) {
        VolumeResult result;
        result.variables = variables;
        result.iterations = iterations;
        
        std::cout << "Testing " << variables << " vars, " << iterations << " iterations... ";
        std::cout.flush();
        
        // Sequential benchmark
        std::cout << "seq... ";
        std::cout.flush();
        result.sequential_ms = run_volume_benchmark(bdd1, bdd2, iterations, false);
        
        if (result.sequential_ms <= 0) {
            std::cout << "❌ Sequential failed" << std::endl;
            continue;
        }
        
        // OpenMP benchmark
        std::cout << "omp... ";
        std::cout.flush();
        result.openmp_ms = run_volume_benchmark(bdd1, bdd2, iterations, true);
        
        if (result.openmp_ms <= 0) {
            std::cout << "❌ OpenMP failed" << std::endl;
            continue;
        }
        
        // Calculate metrics
        result.speedup = (double)result.sequential_ms / result.openmp_ms;
        int num_threads = omp_get_max_threads();
        result.efficiency = (result.speedup / num_threads) * 100.0;  // Efficiency as %
        result.crossover_found = result.speedup > 1.0;
        
        // Print results
        std::cout << std::setw(8) << variables
                  << std::setw(10) << iterations
                  << std::setw(12) << result.sequential_ms << "ms"
                  << std::setw(12) << result.openmp_ms << "ms"
                  << std::setw(10) << std::fixed << std::setprecision(2) << result.speedup << "x"
                  << std::setw(12) << std::setprecision(1) << result.efficiency << "%";
        
        if (result.crossover_found) {
            std::cout << std::setw(10) << "🎯 BENEFIT!";
            if (!found_crossover) {
                std::cout << " ← FIRST!";
                found_crossover = true;
            }
        } else {
            std::cout << std::setw(10) << "No benefit";
        }
        std::cout << std::endl;
        
        results.push_back(result);
        
        // Early success termination if we found good speedup
        if (result.speedup > 1.5) {
            std::cout << "✅ Excellent speedup found, continuing to validate..." << std::endl;
        }
        
        // Safety: stop if tests take too long
        if (result.sequential_ms > 60000) {  // 60 seconds
            std::cout << "\n⚠️  Test taking too long, stopping for safety" << std::endl;
            break;
        }
    }
    
    // Cleanup
    obdd_destroy(bdd1);
    obdd_destroy(bdd2);
    
    // Analysis
    std::cout << "\n=== HIGH-VOLUME CROSSOVER SUMMARY ===" << std::endl;
    
    if (found_crossover) {
        auto first_crossover = std::find_if(results.begin(), results.end(),
            [](const VolumeResult& r) { return r.crossover_found; });
        
        if (first_crossover != results.end()) {
            std::cout << "🎯 CROSSOVER FOUND!" << std::endl;
            std::cout << "   First benefit at: " << first_crossover->iterations 
                      << " iterations (" << first_crossover->variables << " variables)" << std::endl;
            std::cout << "   Speedup: " << first_crossover->speedup << "x" << std::endl;
            std::cout << "   Efficiency: " << first_crossover->efficiency << "%" << std::endl;
            std::cout << "   Sequential time: " << first_crossover->sequential_ms << "ms" << std::endl;
            std::cout << "   OpenMP time: " << first_crossover->openmp_ms << "ms" << std::endl;
            std::cout << "\n💡 PRACTICAL IMPLICATION:" << std::endl;
            std::cout << "   OpenMP becomes beneficial when computation time exceeds " 
                      << first_crossover->sequential_ms << "ms" << std::endl;
        }
        
        // Find best performance
        auto best_speedup = *std::max_element(results.begin(), results.end(),
            [](const VolumeResult& a, const VolumeResult& b) { return a.speedup < b.speedup; });
        
        std::cout << "\n🏆 BEST PERFORMANCE:" << std::endl;
        std::cout << "   Speedup: " << best_speedup.speedup << "x" << std::endl;
        std::cout << "   Efficiency: " << best_speedup.efficiency << "%" << std::endl;
        std::cout << "   At: " << best_speedup.iterations << " iterations" << std::endl;
        
    } else {
        std::cout << "❌ No crossover found in tested range" << std::endl;
        
        if (!results.empty()) {
            auto best_attempt = *std::max_element(results.begin(), results.end(),
                [](const VolumeResult& a, const VolumeResult& b) { return a.speedup < b.speedup; });
            
            std::cout << "   Best speedup: " << best_attempt.speedup << "x at " 
                      << best_attempt.iterations << " iterations" << std::endl;
            std::cout << "   Sequential: " << best_attempt.sequential_ms 
                      << "ms, OpenMP: " << best_attempt.openmp_ms << "ms" << std::endl;
            
            // Calculate what iteration count would be needed
            long overhead = best_attempt.openmp_ms - best_attempt.sequential_ms;
            if (overhead > 0) {
                double needed_ratio = (double)(best_attempt.sequential_ms + overhead) / best_attempt.sequential_ms;
                int needed_iterations = static_cast<int>(best_attempt.iterations * needed_ratio);
                std::cout << "   Estimated iterations needed for crossover: ~" << needed_iterations << std::endl;
            }
        }
    }
    
    save_volume_results();
    
    // Validate data collection
    ASSERT_FALSE(results.empty()) << "No benchmark data collected";
    
    // Check for meaningful computation
    auto longest_test = std::max_element(results.begin(), results.end(),
        [](const VolumeResult& a, const VolumeResult& b) { return a.sequential_ms < b.sequential_ms; });
    
    if (longest_test != results.end() && longest_test->sequential_ms >= 50) {
        std::cout << "\n✅ Achieved meaningful computation times (max: " 
                  << longest_test->sequential_ms << "ms)" << std::endl;
    } else {
        std::cout << "\n⚠️  Computation times still quite small - results may be overhead-dominated" << std::endl;
    }
}