/**
 * @file test_full_scale_timing.cpp
 * @brief Full-scale timing comparison: Sequential vs OpenMP vs CUDA (1K-100K variables)
 * 
 * This test measures actual execution times for all three backends across
 * the full range from 1K to 100K variables with real BDD operations.
 */

#include "core/obdd.hpp"
#include "advanced/obdd_advanced_math.hpp"
#include <gtest/gtest.h>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <vector>
#include <fstream>

#ifdef OBDD_ENABLE_OPENMP
extern "C" {
    OBDDNode* obdd_parallel_apply_omp_enhanced(const OBDD* bdd1, const OBDD* bdd2, OBDD_Op operation);
}
#endif

#ifdef OBDD_ENABLE_CUDA
extern "C" {
    void* obdd_cuda_copy_to_device(const OBDD* bdd);
    void obdd_cuda_and(void* dA, void* dB, void** result);
    void obdd_cuda_or(void* dA, void* dB, void** result); 
    void obdd_cuda_xor(void* dA, void* dB, void** result);
    void obdd_cuda_free_device(void* dHandle);
}
#endif

class FullScaleTimingTest : public ::testing::Test {
protected:
    struct TimingResult {
        int variables;
        double creation_time_ms;
        double sequential_time_ms;
        double openmp_time_ms;
        double cuda_time_ms;
        bool creation_success;
        bool sequential_success;
        bool openmp_success;
        bool cuda_success;
        double openmp_speedup;
        double cuda_speedup;
    };
    
    std::vector<TimingResult> results;
    
    void SetUp() override {
        std::cout << "\n⚡ FULL-SCALE TIMING COMPARISON" << std::endl;
        std::cout << "===============================" << std::endl;
        std::cout << "Testing Sequential vs OpenMP vs CUDA from 1K to 100K variables" << std::endl;
        
#ifdef OBDD_ENABLE_OPENMP
        std::cout << "✅ OpenMP enabled" << std::endl;
#else
        std::cout << "❌ OpenMP disabled" << std::endl;
#endif

#ifdef OBDD_ENABLE_CUDA
        std::cout << "✅ CUDA enabled" << std::endl;
#else
        std::cout << "❌ CUDA disabled" << std::endl;
#endif
        std::cout << std::endl;
    }
    
    void TearDown() override {
        save_results_to_csv();
        print_detailed_analysis();
    }
    
    // Create a more realistic large BDD (but still manageable)
    OBDD* create_realistic_large_bdd(int variables, bool& success) {
        success = false;
        
        std::cout << "    🔨 Creating " << variables << "-variable BDD..." << std::flush;
        
        try {
            std::vector<int> order(variables);
            for (int i = 0; i < variables; ++i) {
                order[i] = i;
            }
            
            OBDD* bdd = obdd_create(variables, order.data());
            if (!bdd) {
                std::cout << " FAILED (structure)" << std::endl;
                return nullptr;
            }
            
            // Create a more complex BDD: alternating OR/AND pattern
            // This creates depth but limits exponential growth
            OBDDNode* result = obdd_constant(1);
            
            // Build in groups to control complexity
            int group_size = std::max(1, std::min(10, variables/100));
            for (int i = 0; i < variables; i += group_size) {
                // Create group: (xi OR xi+1) AND (xi+2 OR xi+3) etc.
                OBDDNode* group_result = obdd_constant(1);
                
                for (int j = 0; j < group_size && (i+j) < variables; j += 2) {
                    if (i+j+1 < variables) {
                        // Create xi OR xi+1
                        OBDD xi_bdd = { nullptr, variables, order.data() };
                        xi_bdd.root = obdd_node_create(i+j, obdd_constant(0), obdd_constant(1));
                        
                        OBDD xi1_bdd = { nullptr, variables, order.data() };
                        xi1_bdd.root = obdd_node_create(i+j+1, obdd_constant(0), obdd_constant(1));
                        
                        OBDDNode* or_result = obdd_apply(&xi_bdd, &xi1_bdd, OBDD_OR);
                        
                        // AND with group result
                        OBDD group_bdd = { group_result, variables, order.data() };
                        OBDD or_bdd = { or_result, variables, order.data() };
                        group_result = obdd_apply(&group_bdd, &or_bdd, OBDD_AND);
                    }
                }
                
                // AND group with overall result
                OBDD result_bdd = { result, variables, order.data() };
                OBDD group_bdd = { group_result, variables, order.data() };
                result = obdd_apply(&result_bdd, &group_bdd, OBDD_AND);
            }
            
            bdd->root = result;
            success = true;
            std::cout << " ✅" << std::endl;
            return bdd;
            
        } catch (...) {
            std::cout << " ❌ (exception)" << std::endl;
            return nullptr;
        }
    }
    
    double measure_operation_time(std::function<bool()> operation) {
        auto start = std::chrono::high_resolution_clock::now();
        bool success = operation();
        auto end = std::chrono::high_resolution_clock::now();
        
        if (!success) {
            return -1.0; // Indicate failure
        }
        
        return std::chrono::duration<double, std::milli>(end - start).count();
    }
    
    void save_results_to_csv() {
        std::ofstream csv("full_scale_timing_results.csv");
        csv << "Variables,Creation_ms,Sequential_ms,OpenMP_ms,CUDA_ms,";
        csv << "Creation_Success,Sequential_Success,OpenMP_Success,CUDA_Success,";
        csv << "OpenMP_Speedup,CUDA_Speedup" << std::endl;
        
        for (const auto& result : results) {
            csv << result.variables << ","
                << result.creation_time_ms << ","
                << result.sequential_time_ms << ","
                << result.openmp_time_ms << ","
                << result.cuda_time_ms << ","
                << (result.creation_success ? 1 : 0) << ","
                << (result.sequential_success ? 1 : 0) << ","
                << (result.openmp_success ? 1 : 0) << ","
                << (result.cuda_success ? 1 : 0) << ","
                << result.openmp_speedup << ","
                << result.cuda_speedup << std::endl;
        }
        
        std::cout << "\n📊 Results saved to: full_scale_timing_results.csv" << std::endl;
    }
    
    void print_detailed_analysis() {
        std::cout << "\n📈 DETAILED TIMING ANALYSIS" << std::endl;
        std::cout << "============================" << std::endl;
        
        std::cout << std::setw(8) << "Vars"
                  << std::setw(12) << "Creation"
                  << std::setw(12) << "Sequential" 
                  << std::setw(12) << "OpenMP"
                  << std::setw(12) << "CUDA"
                  << std::setw(10) << "OMP_Spd"
                  << std::setw(10) << "CUDA_Spd" << std::endl;
        std::cout << std::string(80, '-') << std::endl;
        
        for (const auto& result : results) {
            std::cout << std::setw(8) << result.variables;
            
            // Creation time
            if (result.creation_success) {
                std::cout << std::setw(12) << std::fixed << std::setprecision(3) 
                          << (result.creation_time_ms/1000) << "s";
            } else {
                std::cout << std::setw(12) << "FAILED";
            }
            
            // Sequential time
            if (result.sequential_success) {
                std::cout << std::setw(12) << std::fixed << std::setprecision(3) 
                          << (result.sequential_time_ms/1000) << "s";
            } else {
                std::cout << std::setw(12) << "FAILED";
            }
            
            // OpenMP time
            if (result.openmp_success) {
                std::cout << std::setw(12) << std::fixed << std::setprecision(3) 
                          << (result.openmp_time_ms/1000) << "s";
            } else {
                std::cout << std::setw(12) << "FAILED";
            }
            
            // CUDA time
            if (result.cuda_success) {
                std::cout << std::setw(12) << std::fixed << std::setprecision(3) 
                          << (result.cuda_time_ms/1000) << "s";
            } else {
                std::cout << std::setw(12) << "FAILED";
            }
            
            // Speedups
            std::cout << std::setw(10) << std::fixed << std::setprecision(2) << result.openmp_speedup << "x";
            std::cout << std::setw(10) << std::fixed << std::setprecision(2) << result.cuda_speedup << "x";
            
            std::cout << std::endl;
        }
        
        // Summary statistics
        double best_openmp_speedup = 0;
        double best_cuda_speedup = 0;
        int openmp_wins = 0;
        int cuda_wins = 0;
        
        for (const auto& result : results) {
            if (result.openmp_speedup > best_openmp_speedup) {
                best_openmp_speedup = result.openmp_speedup;
            }
            if (result.cuda_speedup > best_cuda_speedup) {
                best_cuda_speedup = result.cuda_speedup;
            }
            if (result.openmp_speedup > 1.0) openmp_wins++;
            if (result.cuda_speedup > 1.0) cuda_wins++;
        }
        
        std::cout << "\n🏆 PERFORMANCE SUMMARY:" << std::endl;
        std::cout << "Best OpenMP speedup: " << best_openmp_speedup << "x" << std::endl;
        std::cout << "Best CUDA speedup: " << best_cuda_speedup << "x" << std::endl;
        std::cout << "OpenMP wins (>1.0x): " << openmp_wins << "/" << results.size() << std::endl;
        std::cout << "CUDA wins (>1.0x): " << cuda_wins << "/" << results.size() << std::endl;
    }
};

TEST_F(FullScaleTimingTest, ComprehensiveScaleBenchmark) {
    std::cout << "🎯 COMPREHENSIVE SCALE BENCHMARK (1K-100K variables)" << std::endl;
    std::cout << "=====================================================" << std::endl;
    
    // Test scales from 1K to 100K (more aggressive progression)
    std::vector<int> test_scales = {
        1000, 2000, 3000, 5000, 7000, 10000, 
        15000, 20000, 30000, 50000, 75000, 100000
    };
    
    for (int variables : test_scales) {
        std::cout << "\n🔥 Testing " << variables << " variables:" << std::endl;
        
        TimingResult result = {};
        result.variables = variables;
        
        // Step 1: Create BDD
        std::cout << "  📦 Creating BDDs..." << std::endl;
        OBDD* bdd1 = nullptr;
        OBDD* bdd2 = nullptr;
        
        result.creation_time_ms = measure_operation_time([&]() -> bool {
            bdd1 = create_realistic_large_bdd(variables, result.creation_success);
            if (!result.creation_success || !bdd1) return false;
            
            // Create smaller second BDD for operation
            int second_bdd_size = std::min(variables/4, 1000);
            bool success2;
            bdd2 = create_realistic_large_bdd(second_bdd_size, success2);
            
            return success2 && bdd2;
        });
        
        if (!result.creation_success || !bdd1 || !bdd2) {
            std::cout << "  ❌ BDD creation failed, skipping operations" << std::endl;
            if (bdd1) obdd_destroy(bdd1);
            if (bdd2) obdd_destroy(bdd2);
            
            result.sequential_success = false;
            result.openmp_success = false;
            result.cuda_success = false;
            result.openmp_speedup = 0;
            result.cuda_speedup = 0;
            
            results.push_back(result);
            continue;
        }
        
        std::cout << "  ✅ BDD creation: " << std::fixed << std::setprecision(3) 
                  << (result.creation_time_ms/1000) << "s" << std::endl;
        
        // Step 2: Sequential timing
        std::cout << "  🔄 Testing Sequential..." << std::flush;
        volatile OBDDNode* seq_result = nullptr;
        result.sequential_time_ms = measure_operation_time([&]() -> bool {
            seq_result = obdd_apply(bdd1, bdd2, OBDD_AND);
            return seq_result != nullptr;
        });
        
        result.sequential_success = (result.sequential_time_ms >= 0);
        if (result.sequential_success) {
            std::cout << " " << std::fixed << std::setprecision(3) 
                      << (result.sequential_time_ms/1000) << "s ✅" << std::endl;
        } else {
            std::cout << " FAILED ❌" << std::endl;
        }
        
        // Step 3: OpenMP timing
#ifdef OBDD_ENABLE_OPENMP
        std::cout << "  ⚡ Testing OpenMP..." << std::flush;
        volatile OBDDNode* omp_result = nullptr;
        result.openmp_time_ms = measure_operation_time([&]() -> bool {
            omp_result = obdd_parallel_apply_omp_enhanced(bdd1, bdd2, OBDD_AND);
            return omp_result != nullptr;
        });
        
        result.openmp_success = (result.openmp_time_ms >= 0);
        if (result.openmp_success && result.sequential_success) {
            result.openmp_speedup = result.sequential_time_ms / result.openmp_time_ms;
            std::cout << " " << std::fixed << std::setprecision(3) 
                      << (result.openmp_time_ms/1000) << "s (speedup: " 
                      << result.openmp_speedup << "x) ";
            if (result.openmp_speedup > 1.0) {
                std::cout << "🚀" << std::endl;
            } else {
                std::cout << "⚠️" << std::endl;
            }
        } else {
            result.openmp_speedup = 0;
            std::cout << " FAILED ❌" << std::endl;
        }
        
        (void)omp_result; // Suppress warning
#else
        std::cout << "  ⚡ OpenMP not available" << std::endl;
        result.openmp_success = false;
        result.openmp_speedup = 0;
        result.openmp_time_ms = -1;
#endif
        
        // Step 4: CUDA timing
#ifdef OBDD_ENABLE_CUDA
        std::cout << "  🚀 Testing CUDA..." << std::flush;
        volatile void* cuda_result = nullptr;
        result.cuda_time_ms = measure_operation_time([&]() -> bool {
            void* dA = obdd_cuda_copy_to_device(bdd1);
            void* dB = obdd_cuda_copy_to_device(bdd2);
            
            if (!dA || !dB) {
                if (dA) obdd_cuda_free_device(dA);
                if (dB) obdd_cuda_free_device(dB);
                return false;
            }
            
            void* dResult = nullptr;
            obdd_cuda_and(dA, dB, &dResult);
            cuda_result = dResult;
            
            if (dResult) obdd_cuda_free_device(dResult);
            obdd_cuda_free_device(dA);
            obdd_cuda_free_device(dB);
            
            return cuda_result != nullptr;
        });
        
        result.cuda_success = (result.cuda_time_ms >= 0);
        if (result.cuda_success && result.sequential_success) {
            result.cuda_speedup = result.sequential_time_ms / result.cuda_time_ms;
            std::cout << " " << std::fixed << std::setprecision(3) 
                      << (result.cuda_time_ms/1000) << "s (speedup: " 
                      << result.cuda_speedup << "x) ";
            if (result.cuda_speedup > 1.0) {
                std::cout << "🚀" << std::endl;
            } else {
                std::cout << "⚠️" << std::endl;
            }
        } else {
            result.cuda_speedup = 0;
            std::cout << " FAILED ❌" << std::endl;
        }
        
        (void)cuda_result; // Suppress warning
#else
        std::cout << "  🚀 CUDA not available" << std::endl;
        result.cuda_success = false;
        result.cuda_speedup = 0;
        result.cuda_time_ms = -1;
#endif
        
        // Suppress unused warnings
        (void)seq_result;
        
        // Cleanup
        obdd_destroy(bdd1);
        obdd_destroy(bdd2);
        
        results.push_back(result);
        
        std::cout << "  🧹 Cleanup completed" << std::endl;
    }
    
    // Basic test assertions
    EXPECT_GT(results.size(), 0) << "Should have some test results";
    
    // At least some tests should succeed
    bool any_success = false;
    for (const auto& result : results) {
        if (result.creation_success) {
            any_success = true;
            break;
        }
    }
    EXPECT_TRUE(any_success) << "At least some tests should succeed";
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    
    std::cout << "\n⚡ FULL-SCALE TIMING BENCHMARK SUITE" << std::endl;
    std::cout << "====================================" << std::endl;
    std::cout << "Measuring Sequential vs OpenMP vs CUDA from 1K to 100K variables" << std::endl;
    std::cout << "Results will be saved to full_scale_timing_results.csv" << std::endl;
    
    return RUN_ALL_TESTS();
}