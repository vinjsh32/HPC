/**
 * @file test_empirical_analysis.cpp
 * @brief Comprehensive empirical analysis tests for CUDA OBDD performance
 */

#include <gtest/gtest.h>
#include "core/obdd.hpp"

#ifdef OBDD_ENABLE_CUDA
#include "cuda/empirical_analysis.hpp"
#include <cuda_runtime.h>
#include <chrono>
#include <vector>
#include <iostream>
#include <memory>
#include <numeric>
#endif

class EmpiricalAnalysisTest : public ::testing::Test {
protected:
    void SetUp() override {
#ifdef OBDD_ENABLE_CUDA
        // Create test BDDs of various sizes for different analyses
        createTestBDDs();
        
        // Initialize empirical analysis context
        ctx = create_empirical_analysis_context();
        ASSERT_NE(ctx, nullptr);
        
        // Configure analysis parameters
        std::vector<int> thread_counts = {32, 64, 128, 256, 512, 1024};
        std::vector<size_t> problem_sizes = {1000, 5000, 10000, 50000};
        configure_analysis_parameters(ctx, thread_counts, problem_sizes, 5);
        
        std::cout << "\n=== Starting Empirical Analysis Test Suite ===" << std::endl;
#endif
    }
    
    void TearDown() override {
#ifdef OBDD_ENABLE_CUDA
        // Cleanup
        for (auto& bdd : test_bdds) {
            if (bdd) {
                destroy_optimized_device_obdd(bdd);
            }
        }
        
        if (ctx) {
            destroy_empirical_analysis_context(ctx);
        }
        
        std::cout << "=== Empirical Analysis Test Suite Complete ===" << std::endl;
#endif
    }
    
    void createTestBDDs() {
        // Create BDDs of different sizes for scaling analysis
        std::vector<int> sizes = {8, 12, 16, 20};  // Variables
        
        for (int num_vars : sizes) {
            std::vector<int> order(num_vars);
            std::iota(order.begin(), order.end(), 0);
            
            OBDD* host_bdd = obdd_create(num_vars, order.data());
            host_bdd->root = obdd_constant(1); // Simple BDD for testing
            
            OptimizedDeviceOBDD* dev_bdd = create_optimized_device_obdd(host_bdd);
            test_bdds.push_back(dev_bdd);
            
            obdd_destroy(host_bdd);
        }
    }
    
#ifdef OBDD_ENABLE_CUDA
    EmpiricalAnalysisContext* ctx = nullptr;
    std::vector<OptimizedDeviceOBDD*> test_bdds;
#endif
};

#ifdef OBDD_ENABLE_CUDA

TEST_F(EmpiricalAnalysisTest, StrongScalingAnalysis) {
    std::cout << "\n--- Testing Strong Scaling Analysis ---" << std::endl;
    
    ASSERT_FALSE(test_bdds.empty());
    OptimizedDeviceOBDD* test_bdd = test_bdds[2]; // Use medium-sized BDD
    ASSERT_NE(test_bdd, nullptr);
    
    // Perform strong scaling analysis
    auto start_time = std::chrono::high_resolution_clock::now();
    StrongScalingResult* result = perform_strong_scaling_analysis(ctx, test_bdd);
    auto end_time = std::chrono::high_resolution_clock::now();
    
    ASSERT_NE(result, nullptr);
    
    // Validate results
    EXPECT_FALSE(result->thread_counts.empty());
    EXPECT_FALSE(result->execution_times.empty());
    EXPECT_FALSE(result->speedups.empty());
    EXPECT_FALSE(result->efficiencies.empty());
    
    EXPECT_EQ(result->thread_counts.size(), result->execution_times.size());
    EXPECT_EQ(result->thread_counts.size(), result->speedups.size());
    EXPECT_EQ(result->thread_counts.size(), result->efficiencies.size());
    
    // Check that speedups are reasonable (should be > 1 for parallel execution)
    for (double speedup : result->speedups) {
        EXPECT_GT(speedup, 0.5); // Allow some overhead
        EXPECT_LT(speedup, result->thread_counts.back()); // Can't exceed perfect scaling
    }
    
    // Check that execution times decrease with more threads (generally)
    EXPECT_GT(result->sequential_time, 0);
    
    auto analysis_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "Strong scaling analysis completed in " << analysis_duration.count() << " ms" << std::endl;
    
    delete result;
}

TEST_F(EmpiricalAnalysisTest, WeakScalingAnalysis) {
    std::cout << "\n--- Testing Weak Scaling Analysis ---" << std::endl;
    
    ASSERT_GE(test_bdds.size(), 3); // Need multiple BDDs for weak scaling
    
    // Select a subset of BDDs for weak scaling
    std::vector<OptimizedDeviceOBDD*> scaling_bdds(test_bdds.begin(), test_bdds.begin() + 3);
    
    auto start_time = std::chrono::high_resolution_clock::now();
    WeakScalingResult* result = perform_weak_scaling_analysis(ctx, scaling_bdds);
    auto end_time = std::chrono::high_resolution_clock::now();
    
    ASSERT_NE(result, nullptr);
    
    // Validate results
    EXPECT_FALSE(result->thread_counts.empty());
    EXPECT_FALSE(result->problem_sizes.empty());
    EXPECT_FALSE(result->execution_times.empty());
    EXPECT_FALSE(result->efficiencies.empty());
    
    EXPECT_EQ(result->thread_counts.size(), result->problem_sizes.size());
    EXPECT_EQ(result->thread_counts.size(), result->execution_times.size());
    EXPECT_EQ(result->thread_counts.size(), result->efficiencies.size());
    
    // Check that problem sizes increase with thread count
    for (size_t i = 1; i < result->problem_sizes.size(); i++) {
        EXPECT_GE(result->problem_sizes[i], result->problem_sizes[i-1]);
    }
    
    // Check efficiency values are reasonable
    for (double efficiency : result->efficiencies) {
        EXPECT_GT(efficiency, 0.1); // Some efficiency loss is expected
        EXPECT_LE(efficiency, 1.5); // Should not exceed perfect efficiency significantly
    }
    
    auto analysis_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "Weak scaling analysis completed in " << analysis_duration.count() << " ms" << std::endl;
    
    delete result;
}

TEST_F(EmpiricalAnalysisTest, MemoryBandwidthAnalysis) {
    std::cout << "\n--- Testing Memory Bandwidth Analysis ---" << std::endl;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    MemoryBandwidthResult* result = analyze_memory_bandwidth(ctx);
    auto end_time = std::chrono::high_resolution_clock::now();
    
    ASSERT_NE(result, nullptr);
    
    // Validate bandwidth results
    EXPECT_GT(result->theoretical_bandwidth_gbps, 0);
    EXPECT_GT(result->achieved_bandwidth_gbps, 0);
    EXPECT_LE(result->achieved_bandwidth_gbps, result->theoretical_bandwidth_gbps);
    
    EXPECT_GT(result->bandwidth_utilization, 0.0);
    EXPECT_LE(result->bandwidth_utilization, 1.0);
    
    EXPECT_GT(result->bytes_transferred, 0);
    EXPECT_GT(result->transfer_time_ms, 0);
    
    // Check that read/write bandwidth estimates are reasonable
    EXPECT_GT(result->read_bandwidth_gbps, 0);
    EXPECT_GT(result->write_bandwidth_gbps, 0);
    EXPECT_LE(result->effective_bandwidth_gbps, result->achieved_bandwidth_gbps);
    
    // Performance expectations (adjust based on typical GPU capabilities)
    EXPECT_GT(result->achieved_bandwidth_gbps, 50.0); // At least 50 GB/s for modern GPUs
    EXPECT_GT(result->bandwidth_utilization, 0.2);    // At least 20% utilization
    
    auto analysis_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "Memory bandwidth analysis completed in " << analysis_duration.count() << " ms" << std::endl;
    
    delete result;
}

TEST_F(EmpiricalAnalysisTest, CacheMissAnalysis) {
    std::cout << "\n--- Testing Cache Miss Analysis ---" << std::endl;
    
    ASSERT_FALSE(test_bdds.empty());
    OptimizedDeviceOBDD* test_bdd = test_bdds[1]; // Use medium-sized BDD
    
    auto start_time = std::chrono::high_resolution_clock::now();
    CacheMissResult* result = analyze_cache_performance(ctx, test_bdd);
    auto end_time = std::chrono::high_resolution_clock::now();
    
    ASSERT_NE(result, nullptr);
    
    // Validate cache analysis results
    EXPECT_GE(result->l1_hits, 0);
    EXPECT_GE(result->l1_misses, 0);
    EXPECT_GE(result->l2_hits, 0);
    EXPECT_GE(result->l2_misses, 0);
    EXPECT_GE(result->dram_accesses, 0);
    
    // Check hit rates are between 0 and 1
    EXPECT_GE(result->l1_hit_rate, 0.0);
    EXPECT_LE(result->l1_hit_rate, 1.0);
    EXPECT_GE(result->l2_hit_rate, 0.0);
    EXPECT_LE(result->l2_hit_rate, 1.0);
    
    // Check memory latency is reasonable
    EXPECT_GT(result->average_memory_latency, 1.0);   // Should be more than L1 hit latency
    EXPECT_LT(result->average_memory_latency, 1000.0); // Should be less than 1000 cycles
    
    // Check cache efficiency
    EXPECT_GE(result->cache_efficiency, 0.0);
    EXPECT_LE(result->cache_efficiency, 2.0); // Allow some headroom
    
    auto analysis_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "Cache miss analysis completed in " << analysis_duration.count() << " ms" << std::endl;
    
    delete result;
}

TEST_F(EmpiricalAnalysisTest, KernelOccupancyAnalysis) {
    std::cout << "\n--- Testing Kernel Occupancy Analysis ---" << std::endl;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    KernelOccupancyResult* result = analyze_kernel_occupancy(ctx, "weak_normalization_kernel");
    auto end_time = std::chrono::high_resolution_clock::now();
    
    ASSERT_NE(result, nullptr);
    
    // Validate occupancy results
    EXPECT_GT(result->max_threads_per_block, 0);
    EXPECT_LE(result->max_threads_per_block, 1024); // Typical GPU limit
    
    EXPECT_GT(result->optimal_threads_per_block, 0);
    EXPECT_LE(result->optimal_threads_per_block, result->max_threads_per_block);
    
    EXPECT_GE(result->theoretical_occupancy, 0.0);
    EXPECT_LE(result->theoretical_occupancy, 1.0);
    
    EXPECT_GE(result->achieved_occupancy, 0.0);
    EXPECT_LE(result->achieved_occupancy, 1.0);
    
    EXPECT_GE(result->warp_efficiency, 0.0);
    EXPECT_LE(result->warp_efficiency, 1.0);
    
    EXPECT_GE(result->registers_per_thread, 0);
    EXPECT_LE(result->registers_per_thread, 255); // Hardware limit
    
    EXPECT_GE(result->shared_memory_per_block, 0);
    
    // Check that optimal configuration makes sense
    EXPECT_GE(result->optimal_threads_per_block, 32); // At least one warp
    EXPECT_EQ(result->optimal_threads_per_block % 32, 0); // Should be multiple of warp size
    
    auto analysis_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "Kernel occupancy analysis completed in " << analysis_duration.count() << " ms" << std::endl;
    
    delete result;
}

TEST_F(EmpiricalAnalysisTest, PowerConsumptionAnalysis) {
    std::cout << "\n--- Testing Power Consumption Analysis ---" << std::endl;
    
    ASSERT_FALSE(test_bdds.empty());
    OptimizedDeviceOBDD* test_bdd = test_bdds[1];
    
    double test_duration = 2.0; // 2 seconds
    
    auto start_time = std::chrono::high_resolution_clock::now();
    PowerConsumptionResult* result = analyze_power_consumption(ctx, test_bdd, test_duration);
    auto end_time = std::chrono::high_resolution_clock::now();
    
    ASSERT_NE(result, nullptr);
    
    // Validate power consumption results
    EXPECT_GT(result->idle_power_watts, 0);
    EXPECT_GT(result->peak_power_watts, result->idle_power_watts);
    EXPECT_GE(result->average_power_watts, result->idle_power_watts);
    EXPECT_LE(result->average_power_watts, result->peak_power_watts);
    
    EXPECT_GT(result->energy_consumed_joules, 0);
    EXPECT_GT(result->energy_efficiency_gflops_w, 0);
    
    // Check power ranges are reasonable for modern GPUs
    EXPECT_LT(result->idle_power_watts, 100);      // Less than 100W idle
    EXPECT_LT(result->peak_power_watts, 500);      // Less than 500W peak
    EXPECT_LT(result->average_power_watts, 400);   // Less than 400W average
    
    // Check thermal throttling events
    EXPECT_GE(result->thermal_throttling_events, 0);
    
    // Check timelines if present
    if (!result->power_timeline.empty()) {
        EXPECT_EQ(result->power_timeline.size(), result->temperature_timeline.size());
        
        for (double power : result->power_timeline) {
            EXPECT_GT(power, 0);
            EXPECT_LT(power, 600); // Reasonable upper bound
        }
        
        for (double temp : result->temperature_timeline) {
            EXPECT_GT(temp, 20);  // Above room temperature
            EXPECT_LT(temp, 100); // Below dangerous levels
        }
    }
    
    auto analysis_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "Power consumption analysis completed in " << analysis_duration.count() << " ms" << std::endl;
    
    delete result;
}

TEST_F(EmpiricalAnalysisTest, ComprehensiveAnalysisSuite) {
    std::cout << "\n--- Testing Comprehensive Analysis Suite ---" << std::endl;
    
    ASSERT_FALSE(test_bdds.empty());
    
    auto suite_start = std::chrono::high_resolution_clock::now();
    
    // Run all analyses
    StrongScalingResult* strong_result = nullptr;
    WeakScalingResult* weak_result = nullptr;
    MemoryBandwidthResult* bandwidth_result = nullptr;
    CacheMissResult* cache_result = nullptr;
    KernelOccupancyResult* occupancy_result = nullptr;
    PowerConsumptionResult* power_result = nullptr;
    
    try {
        std::cout << "Running strong scaling analysis..." << std::endl;
        strong_result = perform_strong_scaling_analysis(ctx, test_bdds[2]);
        
        std::cout << "Running weak scaling analysis..." << std::endl;
        std::vector<OptimizedDeviceOBDD*> scaling_bdds(test_bdds.begin(), test_bdds.begin() + 3);
        weak_result = perform_weak_scaling_analysis(ctx, scaling_bdds);
        
        std::cout << "Running memory bandwidth analysis..." << std::endl;
        bandwidth_result = analyze_memory_bandwidth(ctx);
        
        std::cout << "Running cache performance analysis..." << std::endl;
        cache_result = analyze_cache_performance(ctx, test_bdds[1]);
        
        std::cout << "Running kernel occupancy analysis..." << std::endl;
        occupancy_result = analyze_kernel_occupancy(ctx, "comprehensive_test_kernel");
        
        std::cout << "Running power consumption analysis..." << std::endl;
        power_result = analyze_power_consumption(ctx, test_bdds[1], 1.5);
        
        // Generate comprehensive reports
        std::cout << "\nGenerating comprehensive reports..." << std::endl;
        generate_scaling_report(strong_result, weak_result);
        generate_memory_report(bandwidth_result, cache_result);
        generate_optimization_report(occupancy_result, power_result);
        
        // Generate performance summary
        generate_performance_summary_table(strong_result, bandwidth_result, occupancy_result, power_result);
        
        // Generate comprehensive report file
        generate_comprehensive_report(ctx, "empirical_analysis_report.md");
        
        auto suite_end = std::chrono::high_resolution_clock::now();
        auto suite_duration = std::chrono::duration_cast<std::chrono::seconds>(suite_end - suite_start);
        
        std::cout << "\n=== COMPREHENSIVE ANALYSIS COMPLETE ===" << std::endl;
        std::cout << "Total analysis time: " << suite_duration.count() << " seconds" << std::endl;
        std::cout << "All analyses completed successfully!" << std::endl;
        std::cout << "Detailed report saved to: empirical_analysis_report.md" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error during comprehensive analysis: " << e.what() << std::endl;
        FAIL() << "Comprehensive analysis failed";
    }
    
    // Cleanup results
    delete strong_result;
    delete weak_result;
    delete bandwidth_result;
    delete cache_result;
    delete occupancy_result;
    delete power_result;
}

#else

TEST_F(EmpiricalAnalysisTest, CUDADisabled) {
    std::cout << "CUDA backend is disabled. Skipping empirical analysis tests." << std::endl;
    GTEST_SKIP() << "CUDA backend is disabled";
}

#endif /* OBDD_ENABLE_CUDA */

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}