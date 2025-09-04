/**
 * @file test_crossover_analysis.cpp
 * @brief Safe progressive testing to find OpenMP and CUDA crossover points
 * 
 * This test incrementally increases BDD complexity with safety checks
 * to identify the exact points where parallelization becomes beneficial.
 */

#include <gtest/gtest.h>
#include "core/obdd.hpp"
#include <chrono>
#include <iostream>
#include <vector>
#include <iomanip>
#include <omp.h>
#include <fstream>
#include <cstdlib>
#include <sys/resource.h>
#include <unistd.h>

class CrossoverAnalysis : public ::testing::Test {
protected:
    struct BenchmarkPoint {
        int variables;
        int bdd_nodes;
        long sequential_ms;
        long openmp_ms;
        long cuda_ms;
        double openmp_speedup;
        double cuda_speedup;
        bool memory_safe;
        bool time_safe;
        size_t memory_used_mb;
    };
    
    std::vector<BenchmarkPoint> results;
    const long MAX_TIME_MS = 30000;  // 30 second safety limit
    const size_t MAX_MEMORY_MB = 2048;  // 2GB memory safety limit
    
    void SetUp() override {
        omp_set_num_threads(std::min(8, omp_get_max_threads()));
        std::cout << "\n🔍 CROSSOVER POINT ANALYSIS" << std::endl;
        std::cout << "Safety limits: " << MAX_TIME_MS << "ms, " << MAX_MEMORY_MB << "MB" << std::endl;
        std::cout << "Thread count: " << omp_get_max_threads() << std::endl;
    }
    
    /**
     * Get current memory usage in MB
     */
    size_t get_memory_usage_mb() {
        struct rusage usage;
        if (getrusage(RUSAGE_SELF, &usage) == 0) {
            // ru_maxrss is in KB on Linux, bytes on macOS
            #ifdef __linux__
            return usage.ru_maxrss / 1024;  // Convert KB to MB
            #else
            return usage.ru_maxrss / (1024 * 1024);  // Convert bytes to MB
            #endif
        }
        return 0;
    }
    
    /**
     * Check if it's safe to continue with larger problems
     */
    bool is_safe_to_continue(long time_ms, size_t memory_mb) {
        if (time_ms > MAX_TIME_MS) {
            std::cout << "⚠️  Time safety limit exceeded (" << time_ms << "ms)" << std::endl;
            return false;
        }
        if (memory_mb > MAX_MEMORY_MB) {
            std::cout << "⚠️  Memory safety limit exceeded (" << memory_mb << "MB)" << std::endl;
            return false;
        }
        return true;
    }
    
    /**
     * Create BDD with controlled complexity
     * Uses a formula that creates exponential growth but remains controllable
     */
    OBDD* create_controlled_complex_bdd(int variables, int complexity_factor = 1) {
        std::vector<int> order(variables);
        for (int i = 0; i < variables; ++i) order[i] = i;
        
        OBDD* bdd = obdd_create(variables, order.data());
        
        // Create a complex but controlled structure
        OBDDNode* false_node = obdd_constant(0);
        OBDDNode* true_node = obdd_constant(1);
        
        // Build layers with increasing complexity
        std::vector<OBDDNode*> current_level = {false_node, true_node};
        
        for (int level = variables - 1; level >= 0; --level) {
            std::vector<OBDDNode*> next_level;
            
            // Controlled expansion - limit nodes per level
            int max_nodes = std::min(8 * complexity_factor, static_cast<int>(current_level.size() * 2));
            
            for (int i = 0; i < max_nodes && i < static_cast<int>(current_level.size()); ++i) {
                for (int j = i + 1; j < static_cast<int>(current_level.size()) && 
                     static_cast<int>(next_level.size()) < max_nodes; ++j) {
                    
                    // Alternate patterns to create meaningful BDD structure
                    OBDDNode* low = (level % 2 == 0) ? current_level[i] : current_level[j];
                    OBDDNode* high = (level % 2 == 0) ? current_level[j] : current_level[i];
                    
                    next_level.push_back(obdd_node_create(level, low, high));
                }
            }
            
            // Fallback if no nodes created
            if (next_level.empty()) {
                next_level.push_back(obdd_node_create(level, current_level[0], 
                    current_level.size() > 1 ? current_level[1] : current_level[0]));
            }
            
            current_level = next_level;
        }
        
        bdd->root = current_level.empty() ? true_node : current_level[0];
        return bdd;
    }
    
    /**
     * Count nodes in BDD safely
     */
    int count_bdd_nodes_safe(OBDD* bdd) {
        if (!bdd || !bdd->root) return 0;
        
        std::set<OBDDNode*> visited;
        std::vector<OBDDNode*> stack = {bdd->root};
        
        while (!stack.empty() && visited.size() < 100000) { // Safety limit
            OBDDNode* node = stack.back();
            stack.pop_back();
            
            if (!node || visited.count(node)) continue;
            visited.insert(node);
            
            if (!is_leaf(node)) {
                if (node->lowChild) stack.push_back(node->lowChild);
                if (node->highChild) stack.push_back(node->highChild);
            }
        }
        
        return static_cast<int>(visited.size());
    }
    
    /**
     * Perform safe timing test
     */
    long time_operation_safe(std::function<OBDDNode*()> operation, const std::string& op_name) {
        auto start = std::chrono::high_resolution_clock::now();
        
        try {
            OBDDNode* result = operation();
            if (!result) {
                std::cout << "  " << op_name << ": NULL result" << std::endl;
                return -1;
            }
        } catch (const std::exception& e) {
            std::cout << "  " << op_name << ": Exception - " << e.what() << std::endl;
            return -1;
        } catch (...) {
            std::cout << "  " << op_name << ": Unknown exception" << std::endl;
            return -1;
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    }
    
    void save_results_to_csv() {
        std::ofstream file("crossover_analysis_results.csv");
        file << "Variables,BDD_Nodes,Sequential_ms,OpenMP_ms,CUDA_ms,OpenMP_Speedup,CUDA_Speedup,Memory_MB,Memory_Safe,Time_Safe\n";
        
        for (const auto& result : results) {
            file << result.variables << ","
                 << result.bdd_nodes << ","
                 << result.sequential_ms << ","
                 << result.openmp_ms << ","
                 << result.cuda_ms << ","
                 << std::fixed << std::setprecision(3) << result.openmp_speedup << ","
                 << result.cuda_speedup << ","
                 << result.memory_used_mb << ","
                 << (result.memory_safe ? "YES" : "NO") << ","
                 << (result.time_safe ? "YES" : "NO") << "\n";
        }
        file.close();
        std::cout << "\n📊 Results saved to: crossover_analysis_results.csv" << std::endl;
    }
};

TEST_F(CrossoverAnalysis, FindCrossoverPoints) {
    std::cout << "\n=== Progressive Crossover Point Analysis ===" << std::endl;
    std::cout << std::setw(8) << "Vars"
              << std::setw(10) << "Nodes"
              << std::setw(12) << "Sequential"
              << std::setw(12) << "OpenMP"
              << std::setw(12) << "CUDA"
              << std::setw(12) << "OMP Speedup"
              << std::setw(12) << "CUDA Speedup"
              << std::setw(10) << "Memory" << std::endl;
    std::cout << std::string(88, '-') << std::endl;
    
    bool found_openmp_crossover = false;
    bool found_cuda_crossover = false;
    
    // Start with smaller problems and increase gradually
    for (int vars = 16; vars <= 40; vars += 2) {
        BenchmarkPoint point;
        point.variables = vars;
        
        std::cout << "Testing " << vars << " variables... ";
        std::cout.flush();
        
        // Create test BDDs with controlled complexity
        int complexity = std::min(vars / 8, 4); // Scale complexity with size
        OBDD* bdd1 = create_controlled_complex_bdd(vars, complexity);
        OBDD* bdd2 = create_controlled_complex_bdd(vars, complexity);
        
        point.bdd_nodes = count_bdd_nodes_safe(bdd1) + count_bdd_nodes_safe(bdd2);
        point.memory_used_mb = get_memory_usage_mb();
        
        std::cout << "(" << point.bdd_nodes << " nodes) ";
        
        // Sequential benchmark
        point.sequential_ms = time_operation_safe([&]() {
            return obdd_apply(bdd1, bdd2, OBDD_AND);
        }, "Sequential");
        
        if (point.sequential_ms < 0) {
            std::cout << "❌ Sequential failed" << std::endl;
            obdd_destroy(bdd1);
            obdd_destroy(bdd2);
            break;
        }
        
        // Safety check after sequential
        point.time_safe = is_safe_to_continue(point.sequential_ms, point.memory_used_mb);
        point.memory_safe = point.memory_used_mb < MAX_MEMORY_MB;
        
        if (!point.time_safe || !point.memory_safe) {
            std::cout << "⚠️  Stopping due to safety limits" << std::endl;
            obdd_destroy(bdd1);
            obdd_destroy(bdd2);
            break;
        }
        
        // OpenMP benchmark
        point.openmp_ms = time_operation_safe([&]() {
            return obdd_parallel_apply_omp(bdd1, bdd2, OBDD_AND);
        }, "OpenMP");
        
        // CUDA benchmark (if available)
        point.cuda_ms = 0;
#ifdef OBDD_ENABLE_CUDA
        point.cuda_ms = time_operation_safe([&]() {
            return obdd_parallel_apply_cuda(bdd1, bdd2, OBDD_AND);
        }, "CUDA");
#endif
        
        // Calculate speedups
        point.openmp_speedup = (point.openmp_ms > 0 && point.sequential_ms > 0) ? 
            (double)point.sequential_ms / point.openmp_ms : 0.0;
        point.cuda_speedup = (point.cuda_ms > 0 && point.sequential_ms > 0) ? 
            (double)point.sequential_ms / point.cuda_ms : 0.0;
        
        // Print results
        std::cout << std::setw(8) << vars
                  << std::setw(10) << point.bdd_nodes
                  << std::setw(12) << point.sequential_ms << "ms"
                  << std::setw(12) << point.openmp_ms << "ms"
                  << std::setw(12) << point.cuda_ms << "ms"
                  << std::setw(12) << std::fixed << std::setprecision(2) 
                  << point.openmp_speedup << "x"
                  << std::setw(12) << point.cuda_speedup << "x"
                  << std::setw(10) << point.memory_used_mb << "MB" << std::endl;
        
        // Check for crossover points
        if (!found_openmp_crossover && point.openmp_speedup > 1.0) {
            std::cout << "🎯 OPENMP CROSSOVER FOUND at " << vars << " variables! Speedup: " 
                      << point.openmp_speedup << "x" << std::endl;
            found_openmp_crossover = true;
        }
        
        if (!found_cuda_crossover && point.cuda_speedup > 1.0) {
            std::cout << "🚀 CUDA CROSSOVER FOUND at " << vars << " variables! Speedup: " 
                      << point.cuda_speedup << "x" << std::endl;
            found_cuda_crossover = true;
        }
        
        results.push_back(point);
        
        // Cleanup
        obdd_destroy(bdd1);
        obdd_destroy(bdd2);
        
        // Early termination if both crossovers found and we're past initial range
        if (found_openmp_crossover && found_cuda_crossover && vars > 24) {
            std::cout << "\n✅ Both crossover points found, stopping analysis" << std::endl;
            break;
        }
        
        // Safety check for next iteration
        if (point.sequential_ms > MAX_TIME_MS / 2) {  // Be conservative
            std::cout << "\n⚠️  Approaching time limit, stopping analysis" << std::endl;
            break;
        }
    }
    
    // Summary
    std::cout << "\n=== CROSSOVER ANALYSIS SUMMARY ===" << std::endl;
    
    if (found_openmp_crossover) {
        auto openmp_point = std::find_if(results.begin(), results.end(),
            [](const BenchmarkPoint& p) { return p.openmp_speedup > 1.0; });
        if (openmp_point != results.end()) {
            std::cout << "✅ OpenMP becomes beneficial at: " << openmp_point->variables 
                      << " variables (" << openmp_point->openmp_speedup << "x speedup)" << std::endl;
        }
    } else {
        std::cout << "❌ OpenMP crossover point not found in tested range" << std::endl;
        if (!results.empty()) {
            auto best_openmp = *std::max_element(results.begin(), results.end(),
                [](const BenchmarkPoint& a, const BenchmarkPoint& b) {
                    return a.openmp_speedup < b.openmp_speedup;
                });
            std::cout << "   Best OpenMP speedup: " << best_openmp.openmp_speedup 
                      << "x at " << best_openmp.variables << " variables" << std::endl;
        }
    }
    
    if (found_cuda_crossover) {
        auto cuda_point = std::find_if(results.begin(), results.end(),
            [](const BenchmarkPoint& p) { return p.cuda_speedup > 1.0; });
        if (cuda_point != results.end()) {
            std::cout << "✅ CUDA becomes beneficial at: " << cuda_point->variables 
                      << " variables (" << cuda_point->cuda_speedup << "x speedup)" << std::endl;
        }
    } else {
        std::cout << "❌ CUDA crossover point not found in tested range" << std::endl;
        if (!results.empty()) {
            auto best_cuda = *std::max_element(results.begin(), results.end(),
                [](const BenchmarkPoint& a, const BenchmarkPoint& b) {
                    return a.cuda_speedup < b.cuda_speedup;
                });
            std::cout << "   Best CUDA speedup: " << best_cuda.cuda_speedup 
                      << "x at " << best_cuda.variables << " variables" << std::endl;
        }
    }
    
    save_results_to_csv();
    
    // Validate we found meaningful results
    ASSERT_FALSE(results.empty()) << "No benchmark data collected";
    ASSERT_TRUE(results.size() >= 3) << "Insufficient data points for analysis";
}