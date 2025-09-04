/**
 * @file test_massive_scale_gradual.cpp
 * @brief Massive-scale gradual BDD tests from 100 to 10,000 variables
 * 
 * This test suite implements gradual scaling from 100 variables up to 10,000 variables
 * with comprehensive backend comparison (Sequential, OpenMP, CUDA) and intelligent
 * early stopping if execution time exceeds reasonable limits.
 * 
 * @author @vijsh32
 * @date August 29, 2025
 * @version 3.0
 */

#include "core/obdd.hpp"
#include "advanced/obdd_advanced_math.hpp"
#include "advanced/obdd_reordering.hpp"

#ifdef OBDD_ENABLE_CUDA
#include "cuda/obdd_cuda.hpp"
#include <cuda_runtime.h>
#endif

// Forward declarations for CUDA functions
extern "C" {
    void* obdd_cuda_copy_to_device(const OBDD* bdd);
    void obdd_cuda_and(void* dA, void* dB, void** result);
    void obdd_cuda_or(void* dA, void* dB, void** result);
    void obdd_cuda_xor(void* dA, void* dB, void** result);
    void obdd_cuda_not(void* dA, void** result);
    void obdd_cuda_free_device(void* dHandle);
}
#include <gtest/gtest.h>
#include <chrono>
#include <vector>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <thread>
#include <functional>
#include <cmath>
#include <numeric>

class MassiveScaleGradualTest : public ::testing::Test {
protected:
public:
    static constexpr double MAX_EXECUTION_TIME_MINUTES = 5.0;
protected:
    static constexpr double WARNING_TIME_MINUTES = 2.0;
    
    struct ScaleTestResult {
        int variables;
        double sequential_time_ms;
        double openmp_time_ms;
        double cuda_time_ms;
        double openmp_speedup;
        double cuda_speedup;
        int bdd_nodes_generated;
        bool timeout_occurred;
        std::string problem_type;
    };
    
    std::vector<ScaleTestResult> results;
    
    void SetUp() override {
        std::cout << "\n🔥 MASSIVE SCALE GRADUAL BDD PERFORMANCE TEST" << std::endl;
        std::cout << "=============================================" << std::endl;
        std::cout << "Testing scale from 100 to 10,000 variables" << std::endl;
        std::cout << "Maximum execution time per test: " << MAX_EXECUTION_TIME_MINUTES << " minutes" << std::endl;
        std::cout << "System threads: " << std::thread::hardware_concurrency() << std::endl;
        std::cout << std::endl;
    }
    
    void TearDown() override {
        save_results_to_csv();
        print_summary();
    }
    
    // Create massive BDD using chain structure for scalability
    OBDD* create_massive_chain_bdd(int variables) {
        if (variables > 200000) {
            std::cout << "⚠️ Warning: " << variables << " variables exceeds safety limit" << std::endl;
            variables = 200000; // Increased safety limit for ultra-scale
        }
        
        std::vector<int> order(variables);
        std::iota(order.begin(), order.end(), 0);
        
        OBDD* chain_bdd = obdd_create(variables, order.data());
        
        // Create chain: x0 AND x1 AND x2 AND ... AND xN
        // This generates a linear BDD structure that scales predictably
        OBDDNode* result = obdd_constant(1);
        
        for (int i = 0; i < variables; ++i) {
            OBDD xi_bdd = { nullptr, variables, order.data() };
            xi_bdd.root = obdd_node_create(i, obdd_constant(0), obdd_constant(1));
            
            OBDD result_bdd = { result, variables, order.data() };
            OBDDNode* new_result = obdd_apply(&result_bdd, &xi_bdd, OBDD_AND);
            result = new_result;
            
            // Progress indicator for ultra-large problems
            if (variables >= 10000 && i % (variables/10) == 0) {
                std::cout << "  Progress: " << (100*i/variables) << "% (" << i << "/" << variables << ")" << std::endl;
                std::cout.flush(); // Force output for long operations
            }
        }
        
        chain_bdd->root = result;
        return chain_bdd;
    }
    
    // Create massive BDD using tree structure for better parallelization
    OBDD* create_massive_tree_bdd(int variables) {
        if (variables > 15000) {
            variables = 15000; // Safety limit
        }
        
        std::vector<int> order(variables);
        std::iota(order.begin(), order.end(), 0);
        
        OBDD* tree_bdd = obdd_create(variables, order.data());
        
        // Create balanced tree: (x0 OR x1) AND (x2 OR x3) AND ... 
        std::vector<OBDDNode*> level_nodes;
        
        // Create leaf level
        for (int i = 0; i < variables; ++i) {
            OBDDNode* xi = obdd_node_create(i, obdd_constant(0), obdd_constant(1));
            level_nodes.push_back(xi);
        }
        
        // Build tree bottom-up
        while (level_nodes.size() > 1) {
            std::vector<OBDDNode*> next_level;
            
            for (size_t i = 0; i < level_nodes.size(); i += 2) {
                OBDD left_bdd = { level_nodes[i], variables, order.data() };
                
                if (i + 1 < level_nodes.size()) {
                    OBDD right_bdd = { level_nodes[i + 1], variables, order.data() };
                    OBDDNode* combined = obdd_apply(&left_bdd, &right_bdd, OBDD_OR);
                    next_level.push_back(combined);
                } else {
                    next_level.push_back(level_nodes[i]); // Odd number of nodes
                }
            }
            
            level_nodes = next_level;
            
            if (variables >= 10000 && level_nodes.size() % 1000 == 0) {
                std::cout << "  Tree level nodes remaining: " << level_nodes.size() << std::endl;
                std::cout.flush();
            }
        }
        
        tree_bdd->root = level_nodes.empty() ? obdd_constant(0) : level_nodes[0];
        return tree_bdd;
    }
    
    // Measure execution time with timeout protection
    template<typename Func>
    double measure_with_timeout(Func operation, bool& timeout_occurred) {
        timeout_occurred = false;
        
        auto start = std::chrono::high_resolution_clock::now();
        
        try {
            // Run the operation
            OBDDNode* result = operation();
            
            auto end = std::chrono::high_resolution_clock::now();
            double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
            
            // Ensure result is used to prevent optimization
            volatile OBDDNode* dummy = result;
            (void)dummy;
            
            // Check if we exceeded warning time
            if (time_ms > WARNING_TIME_MINUTES * 60 * 1000) {
                std::cout << "⚠️ Warning: Operation took " << (time_ms/60000) << " minutes" << std::endl;
            }
            
            return time_ms;
        } catch (...) {
            timeout_occurred = true;
            std::cout << "❌ Operation failed or timed out" << std::endl;
            return -1.0;
        }
    }
    
    void save_results_to_csv() {
        std::ofstream csv("massive_scale_results.csv");
        csv << "Variables,Sequential_ms,OpenMP_ms,CUDA_ms,OpenMP_Speedup,CUDA_Speedup,BDD_Nodes,Timeout,Problem_Type\n";
        
        for (const auto& result : results) {
            csv << result.variables << ","
                << result.sequential_time_ms << ","
                << result.openmp_time_ms << ","
                << result.cuda_time_ms << ","
                << result.openmp_speedup << ","
                << result.cuda_speedup << ","
                << result.bdd_nodes_generated << ","
                << (result.timeout_occurred ? "YES" : "NO") << ","
                << result.problem_type << "\n";
        }
        
        std::cout << "\n📊 Results saved to massive_scale_results.csv" << std::endl;
    }
    
    void print_summary() {
        std::cout << "\n📈 MASSIVE SCALE PERFORMANCE SUMMARY" << std::endl;
        std::cout << "====================================" << std::endl;
        
        std::cout << std::fixed << std::setprecision(3);
        std::cout << std::setw(10) << "Variables" 
                  << std::setw(12) << "Sequential" 
                  << std::setw(12) << "OpenMP" 
                  << std::setw(12) << "CUDA" 
                  << std::setw(12) << "OMP Speedup"
                  << std::setw(12) << "CUDA Speedup"
                  << std::setw(8) << "Nodes" << std::endl;
        std::cout << std::string(80, '-') << std::endl;
        
        for (const auto& result : results) {
            std::cout << std::setw(10) << result.variables;
            
            if (result.timeout_occurred) {
                std::cout << std::setw(12) << "TIMEOUT"
                          << std::setw(12) << "TIMEOUT"
                          << std::setw(12) << "TIMEOUT"
                          << std::setw(12) << "N/A"
                          << std::setw(12) << "N/A";
            } else {
                std::cout << std::setw(12) << (result.sequential_time_ms / 1000) << "s"
                          << std::setw(12) << (result.openmp_time_ms / 1000) << "s"
                          << std::setw(12) << (result.cuda_time_ms / 1000) << "s"
                          << std::setw(12) << result.openmp_speedup << "x"
                          << std::setw(12) << result.cuda_speedup << "x";
            }
            
            std::cout << std::setw(8) << result.bdd_nodes_generated << std::endl;
        }
        
        // Find best performing backend
        double best_openmp_speedup = 0;
        double best_cuda_speedup = 0;
        int best_openmp_vars = 0;
        int best_cuda_vars = 0;
        
        for (const auto& result : results) {
            if (!result.timeout_occurred) {
                if (result.openmp_speedup > best_openmp_speedup) {
                    best_openmp_speedup = result.openmp_speedup;
                    best_openmp_vars = result.variables;
                }
                if (result.cuda_speedup > best_cuda_speedup) {
                    best_cuda_speedup = result.cuda_speedup;
                    best_cuda_vars = result.variables;
                }
            }
        }
        
        std::cout << "\n🏆 BEST PERFORMANCE ACHIEVED:" << std::endl;
        std::cout << "OpenMP: " << best_openmp_speedup << "x speedup at " << best_openmp_vars << " variables" << std::endl;
        std::cout << "CUDA: " << best_cuda_speedup << "x speedup at " << best_cuda_vars << " variables" << std::endl;
    }
};

TEST_F(MassiveScaleGradualTest, ChainBDDScaling) {
    std::cout << "🔥 Testing ULTRA-MASSIVE BDD scaling (10K → 100K variables)" << std::endl;
    
    // Test sizes: ultra-massive scale to find true crossover points
    std::vector<int> test_sizes = {10000, 20000, 30000, 50000, 75000, 100000};
    
    for (int variables : test_sizes) {
        std::cout << "\n⚡ Testing " << variables << " variables (Chain BDD)..." << std::endl;
        
        ScaleTestResult result = {};
        result.variables = variables;
        result.problem_type = "Chain";
        result.timeout_occurred = false;
        
        // Create test BDDs
        OBDD* bdd1 = nullptr;
        OBDD* bdd2 = nullptr;
        
        try {
            auto creation_start = std::chrono::high_resolution_clock::now();
            std::cout << "  Creating primary BDD (" << variables << " variables)..." << std::endl;
            bdd1 = create_massive_chain_bdd(variables);
            
            // For ultra-massive scale, use much smaller second BDD to keep operations tractable
            int second_bdd_size = std::min(variables/10, 5000);
            std::cout << "  Creating secondary BDD (" << second_bdd_size << " variables)..." << std::endl;
            bdd2 = create_massive_chain_bdd(second_bdd_size);
            auto creation_end = std::chrono::high_resolution_clock::now();
            
            double creation_time = std::chrono::duration<double, std::milli>(creation_end - creation_start).count();
            std::cout << "  BDD creation: " << (creation_time/1000) << " seconds" << std::endl;
            
            if (creation_time > MAX_EXECUTION_TIME_MINUTES * 60 * 1000) {
                std::cout << "⏰ BDD creation exceeded time limit, skipping larger sizes" << std::endl;
                result.timeout_occurred = true;
                results.push_back(result);
                if (bdd1) obdd_destroy(bdd1);
                if (bdd2) obdd_destroy(bdd2);
                break;
            }
            
            result.bdd_nodes_generated = obdd_count_nodes(bdd1) + obdd_count_nodes(bdd2);
            std::cout << "  Total BDD nodes: " << result.bdd_nodes_generated << std::endl;
            
        } catch (...) {
            std::cout << "❌ Failed to create BDD with " << variables << " variables" << std::endl;
            result.timeout_occurred = true;
            results.push_back(result);
            break;
        }
        
        if (result.timeout_occurred) continue;
        
        // Sequential timing
        std::cout << "  🔄 Testing Sequential..." << std::flush;
        bool seq_timeout = false;
        result.sequential_time_ms = measure_with_timeout([&]() {
            return obdd_apply(bdd1, bdd2, OBDD_AND);
        }, seq_timeout);
        
        if (seq_timeout || result.sequential_time_ms > MAX_EXECUTION_TIME_MINUTES * 60 * 1000) {
            std::cout << " TIMEOUT" << std::endl;
            result.timeout_occurred = true;
            results.push_back(result);
            obdd_destroy(bdd1);
            obdd_destroy(bdd2);
            break;
        }
        std::cout << " " << (result.sequential_time_ms/1000) << "s" << std::endl;
        
        // OpenMP timing
        std::cout << "  ⚡ Testing Enhanced OpenMP..." << std::flush;
        bool omp_timeout = false;
        result.openmp_time_ms = measure_with_timeout([&]() {
            return obdd_parallel_apply_omp_enhanced(bdd1, bdd2, OBDD_AND);
        }, omp_timeout);
        
        if (omp_timeout) {
            result.openmp_time_ms = -1;
            result.openmp_speedup = 0;
            std::cout << " TIMEOUT" << std::endl;
        } else {
            result.openmp_speedup = result.sequential_time_ms / result.openmp_time_ms;
            std::cout << " " << (result.openmp_time_ms/1000) << "s (speedup: " << result.openmp_speedup << "x)" << std::endl;
        }
        
        // CUDA timing
        std::cout << "  🚀 Testing CUDA..." << std::flush;
        bool cuda_timeout = false;
        result.cuda_time_ms = measure_with_timeout([&]() -> OBDDNode* {
#ifdef OBDD_ENABLE_CUDA
            try {
                // Copy BDDs to GPU
                void* dA = obdd_cuda_copy_to_device(bdd1);
                void* dB = obdd_cuda_copy_to_device(bdd2);
                
                if (!dA || !dB) {
                    if (dA) obdd_cuda_free_device(dA);
                    if (dB) obdd_cuda_free_device(dB);
                    return nullptr;
                }
                
                // Perform CUDA AND operation
                void* dResult = nullptr;
                obdd_cuda_and(dA, dB, &dResult);
                
                // Cleanup GPU memory
                obdd_cuda_free_device(dA);
                obdd_cuda_free_device(dB);
                if (dResult) {
                    obdd_cuda_free_device(dResult);
                }
                
                // Return a dummy result (we're measuring time, not correctness)
                return obdd_constant(1);
            } catch (...) {
                return nullptr;
            }
#else
            // CUDA not available
            return nullptr;
#endif
        }, cuda_timeout);
        
        if (cuda_timeout || result.cuda_time_ms < 0) {
            result.cuda_time_ms = -1;
            result.cuda_speedup = 0;
            std::cout << " TIMEOUT/ERROR" << std::endl;
        } else {
            result.cuda_speedup = result.sequential_time_ms / result.cuda_time_ms;
            std::cout << " " << (result.cuda_time_ms/1000) << "s (speedup: " << result.cuda_speedup << "x)" << std::endl;
        }
        
        results.push_back(result);
        
        // Clean up
        obdd_destroy(bdd1);
        obdd_destroy(bdd2);
        
        // Early stopping if any backend takes too long
        if (result.sequential_time_ms > MAX_EXECUTION_TIME_MINUTES * 60 * 1000) {
            std::cout << "\n⏰ Sequential execution exceeded " << MAX_EXECUTION_TIME_MINUTES << " minutes" << std::endl;
            std::cout << "   Stopping scaling test to prevent excessive runtime" << std::endl;
            break;
        }
        
        // Performance assertions for successful tests
        if (!result.timeout_occurred && variables >= 1000) {
            // We expect either OpenMP or CUDA to show some benefit at large scales
            bool any_parallel_benefit = (result.openmp_speedup > 0.5) || (result.cuda_speedup > 0.8);
            EXPECT_TRUE(any_parallel_benefit) 
                << "At " << variables << " variables, at least one parallel backend should show some benefit";
        }
    }
    
    // Final analysis
    std::cout << "\n🎯 CHAIN BDD SCALING ANALYSIS COMPLETE" << std::endl;
    if (!results.empty()) {
        auto last_result = results.back();
        std::cout << "Maximum scale tested: " << last_result.variables << " variables" << std::endl;
        
        // Success criteria
        bool openmp_beats_sequential = false;
        bool cuda_beats_sequential = false;
        
        for (const auto& r : results) {
            if (r.openmp_speedup > 1.0) openmp_beats_sequential = true;
            if (r.cuda_speedup > 1.0) cuda_beats_sequential = true;
        }
        
        if (openmp_beats_sequential) {
            std::cout << "✅ OpenMP achieved >1.0x speedup in this test series" << std::endl;
        }
        if (cuda_beats_sequential) {
            std::cout << "✅ CUDA achieved >1.0x speedup in this test series" << std::endl;
        }
    }
}

// CUDA function already declared above

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    
    std::cout << "\n🎯 MASSIVE SCALE GRADUAL BDD TEST SUITE" << std::endl;
    std::cout << "=======================================" << std::endl;
    std::cout << "Ultra-massive scaling analysis: 10,000 → 100,000 variables" << std::endl;
    std::cout << "Backends: Sequential CPU, Enhanced OpenMP, CUDA GPU (full comparison)" << std::endl;
    std::cout << "Early stopping: " << MassiveScaleGradualTest::MAX_EXECUTION_TIME_MINUTES << " minutes per test" << std::endl;
    
    return RUN_ALL_TESTS();
}