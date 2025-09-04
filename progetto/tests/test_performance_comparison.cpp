/**
 * @file test_performance_comparison.cpp
 * @brief Simple performance comparison between Sequential, OpenMP, and CUDA backends
 * 
 * This test provides clean comparison without memory issues by using
 * manageable problem sizes and proper backend detection.
 */

#include "core/obdd.hpp"
#include "advanced/obdd_advanced_math.hpp"
#include <gtest/gtest.h>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <vector>

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

class PerformanceComparisonTest : public ::testing::Test {
protected:
    void SetUp() override {
        std::cout << "\n⚡ BACKEND PERFORMANCE COMPARISON" << std::endl;
        std::cout << "=================================" << std::endl;
        
        // Show available backends
        std::cout << "Available backends:" << std::endl;
        std::cout << "- Sequential CPU: ✅ Always available" << std::endl;
        
#ifdef OBDD_ENABLE_OPENMP
        std::cout << "- OpenMP Parallel: ✅ Enabled" << std::endl;
#else
        std::cout << "- OpenMP Parallel: ❌ Disabled" << std::endl;
#endif

#ifdef OBDD_ENABLE_CUDA
        std::cout << "- CUDA GPU: ✅ Enabled" << std::endl;
#else
        std::cout << "- CUDA GPU: ❌ Disabled" << std::endl;
#endif
        std::cout << std::endl;
    }
    
    // Create a moderately complex BDD for testing
    OBDD* create_test_bdd(int variables) {
        std::vector<int> order(variables);
        for (int i = 0; i < variables; ++i) {
            order[i] = i;
        }
        
        OBDD* bdd = obdd_create(variables, order.data());
        
        // Create a chain of AND operations: x0 AND x1 AND x2 AND ...
        OBDDNode* result = obdd_constant(1);
        
        for (int i = 0; i < variables; ++i) {
            OBDD xi_bdd = { nullptr, variables, order.data() };
            xi_bdd.root = obdd_node_create(i, obdd_constant(0), obdd_constant(1));
            
            OBDD result_bdd = { result, variables, order.data() };
            result = obdd_apply(&result_bdd, &xi_bdd, OBDD_AND);
        }
        
        bdd->root = result;
        return bdd;
    }
    
    double measure_time_ms(std::function<void()> operation) {
        auto start = std::chrono::high_resolution_clock::now();
        operation();
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start).count();
    }
};

TEST_F(PerformanceComparisonTest, SmallScaleComparison) {
    std::cout << "🔹 SMALL SCALE COMPARISON (6-10 variables)" << std::endl;
    std::cout << "===========================================" << std::endl;
    
    for (int vars = 6; vars <= 10; vars += 2) {
        std::cout << "\nTesting " << vars << " variables..." << std::endl;
        
        OBDD* bdd1 = create_test_bdd(vars);
        OBDD* bdd2 = create_test_bdd(vars);
        
        if (!bdd1 || !bdd2) {
            std::cout << "❌ Failed to create test BDDs" << std::endl;
            continue;
        }
        
        // Sequential timing
        volatile OBDDNode* seq_result = nullptr;
        double seq_time = measure_time_ms([&]() {
            seq_result = obdd_apply(bdd1, bdd2, OBDD_AND);
        });
        
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "  Sequential: " << std::setw(8) << seq_time << " ms" << std::endl;
        
#ifdef OBDD_ENABLE_OPENMP
        // OpenMP timing
        volatile OBDDNode* omp_result = nullptr;
        double omp_time = measure_time_ms([&]() {
            omp_result = obdd_parallel_apply_omp_enhanced(bdd1, bdd2, OBDD_AND);
        });
        
        double omp_speedup = seq_time / omp_time;
        std::cout << "  OpenMP:     " << std::setw(8) << omp_time << " ms (speedup: " 
                  << std::setw(5) << omp_speedup << "x)" << std::endl;
#endif

#ifdef OBDD_ENABLE_CUDA
        // CUDA timing
        volatile void* cuda_result = nullptr;
        double cuda_time = measure_time_ms([&]() {
            void* dA = obdd_cuda_copy_to_device(bdd1);
            void* dB = obdd_cuda_copy_to_device(bdd2);
            
            if (dA && dB) {
                void* dResult = nullptr;
                obdd_cuda_and(dA, dB, &dResult);
                cuda_result = dResult;
                
                if (dResult) obdd_cuda_free_device(dResult);
            }
            
            if (dA) obdd_cuda_free_device(dA);
            if (dB) obdd_cuda_free_device(dB);
        });
        
        double cuda_speedup = seq_time / cuda_time;
        std::cout << "  CUDA:       " << std::setw(8) << cuda_time << " ms (speedup: " 
                  << std::setw(5) << cuda_speedup << "x)" << std::endl;
#endif
        
        obdd_destroy(bdd1);
        obdd_destroy(bdd2);
        
        // Suppress unused variable warnings
        (void)seq_result;
#ifdef OBDD_ENABLE_OPENMP
        (void)omp_result;
#endif
#ifdef OBDD_ENABLE_CUDA
        (void)cuda_result;
#endif
    }
}

TEST_F(PerformanceComparisonTest, MediumScaleComparison) {
    std::cout << "\n🔹 MEDIUM SCALE COMPARISON (12-16 variables)" << std::endl;
    std::cout << "=============================================" << std::endl;
    
    for (int vars = 12; vars <= 16; vars += 2) {
        std::cout << "\nTesting " << vars << " variables..." << std::endl;
        
        OBDD* bdd1 = create_test_bdd(vars);
        OBDD* bdd2 = create_test_bdd(vars);
        
        if (!bdd1 || !bdd2) {
            std::cout << "❌ Failed to create test BDDs" << std::endl;
            continue;
        }
        
        // Sequential timing
        volatile OBDDNode* seq_result = nullptr;
        double seq_time = measure_time_ms([&]() {
            seq_result = obdd_apply(bdd1, bdd2, OBDD_AND);
        });
        
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "  Sequential: " << std::setw(8) << seq_time << " ms" << std::endl;
        
#ifdef OBDD_ENABLE_OPENMP
        // OpenMP timing
        volatile OBDDNode* omp_result = nullptr;
        double omp_time = measure_time_ms([&]() {
            omp_result = obdd_parallel_apply_omp_enhanced(bdd1, bdd2, OBDD_AND);
        });
        
        double omp_speedup = seq_time / omp_time;
        std::cout << "  OpenMP:     " << std::setw(8) << omp_time << " ms (speedup: " 
                  << std::setw(5) << omp_speedup << "x)" << std::endl;
#endif

#ifdef OBDD_ENABLE_CUDA
        // CUDA timing  
        volatile void* cuda_result = nullptr;
        double cuda_time = measure_time_ms([&]() {
            void* dA = obdd_cuda_copy_to_device(bdd1);
            void* dB = obdd_cuda_copy_to_device(bdd2);
            
            if (dA && dB) {
                void* dResult = nullptr;
                obdd_cuda_and(dA, dB, &dResult);
                cuda_result = dResult;
                
                if (dResult) obdd_cuda_free_device(dResult);
            }
            
            if (dA) obdd_cuda_free_device(dA);
            if (dB) obdd_cuda_free_device(dB);
        });
        
        double cuda_speedup = seq_time / cuda_time;
        std::cout << "  CUDA:       " << std::setw(8) << cuda_time << " ms (speedup: " 
                  << std::setw(5) << cuda_speedup << "x)" << std::endl;
#endif
        
        obdd_destroy(bdd1);
        obdd_destroy(bdd2);
        
        // Suppress unused variable warnings
        (void)seq_result;
#ifdef OBDD_ENABLE_OPENMP
        (void)omp_result;
#endif
#ifdef OBDD_ENABLE_CUDA
        (void)cuda_result;
#endif
    }
}

TEST_F(PerformanceComparisonTest, AdvancedMathComparison) {
    std::cout << "\n🔹 ADVANCED MATH COMPARISON" << std::endl;
    std::cout << "===========================" << std::endl;
    
    // Test with mathematical functions
    std::cout << "\nTesting AES S-box problem..." << std::endl;
    
    OBDD* aes_bdd = obdd_aes_sbox();
    if (!aes_bdd) {
        GTEST_SKIP() << "AES S-box BDD creation failed";
    }
    
    // Create a smaller second BDD for comparison
    OBDD* test_bdd = create_test_bdd(8);
    if (!test_bdd) {
        obdd_destroy(aes_bdd);
        GTEST_SKIP() << "Test BDD creation failed";
    }
    
    std::cout << "AES S-box BDD: " << aes_bdd->numVars << " variables" << std::endl;
    std::cout << "Test BDD: " << test_bdd->numVars << " variables" << std::endl;
    
    // Sequential timing
    volatile OBDDNode* seq_result = nullptr;
    double seq_time = measure_time_ms([&]() {
        seq_result = obdd_apply(aes_bdd, test_bdd, OBDD_XOR);
    });
    
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "  Sequential: " << std::setw(8) << seq_time << " ms" << std::endl;
    
#ifdef OBDD_ENABLE_OPENMP
    // OpenMP timing
    volatile OBDDNode* omp_result = nullptr;
    double omp_time = measure_time_ms([&]() {
        omp_result = obdd_parallel_apply_omp_enhanced(aes_bdd, test_bdd, OBDD_XOR);
    });
    
    double omp_speedup = seq_time / omp_time;
    std::cout << "  OpenMP:     " << std::setw(8) << omp_time << " ms (speedup: " 
              << std::setw(5) << omp_speedup << "x)" << std::endl;
#endif

#ifdef OBDD_ENABLE_CUDA
    // CUDA timing
    volatile void* cuda_result = nullptr;
    double cuda_time = measure_time_ms([&]() {
        void* dA = obdd_cuda_copy_to_device(aes_bdd);
        void* dB = obdd_cuda_copy_to_device(test_bdd);
        
        if (dA && dB) {
            void* dResult = nullptr;
            obdd_cuda_xor(dA, dB, &dResult);
            cuda_result = dResult;
            
            if (dResult) obdd_cuda_free_device(dResult);
        }
        
        if (dA) obdd_cuda_free_device(dA);
        if (dB) obdd_cuda_free_device(dB);
    });
    
    double cuda_speedup = seq_time / cuda_time;
    std::cout << "  CUDA:       " << std::setw(8) << cuda_time << " ms (speedup: " 
              << std::setw(5) << cuda_speedup << "x)" << std::endl;
#endif
    
    obdd_destroy(aes_bdd);
    obdd_destroy(test_bdd);
    
    // Suppress unused variable warnings
    (void)seq_result;
#ifdef OBDD_ENABLE_OPENMP
    (void)omp_result;
#endif
#ifdef OBDD_ENABLE_CUDA
    (void)cuda_result;
#endif
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    
    std::cout << "\n⚡ OBDD BACKEND PERFORMANCE COMPARISON" << std::endl;
    std::cout << "======================================" << std::endl;
    std::cout << "Comparing Sequential CPU, OpenMP Parallel, and CUDA GPU performance" << std::endl;
    
    return RUN_ALL_TESTS();
}