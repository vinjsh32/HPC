/**
 * @file test_performance.cpp
 * @brief Performance tests for CUDA optimizations
 */

#include <gtest/gtest.h>
#include "obdd.hpp"

#ifdef OBDD_ENABLE_CUDA
#include "obdd_cuda_optimized.cuh"
#include <cuda_runtime.h>
#include <chrono>
#include <vector>
#include <iostream>
#endif

class PerformanceTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create test BDD with multiple variables
        int order[] = {0, 1, 2, 3, 4, 5, 6, 7};
        test_bdd = obdd_create(8, order);
        
        // Build simple test BDD structure - just create constants
        test_bdd->root = obdd_constant(1);  // TRUE terminal
    }
    
    void TearDown() override {
        if (test_bdd) {
            obdd_destroy(test_bdd);
        }
    }
    
    OBDD* test_bdd;
};

#ifdef OBDD_ENABLE_CUDA

TEST_F(PerformanceTest, OptimizedDeviceOBDDCreation) {
    auto start = std::chrono::high_resolution_clock::now();
    
    OptimizedDeviceOBDD* dev_bdd = create_optimized_device_obdd(test_bdd);
    ASSERT_NE(dev_bdd, nullptr);
    EXPECT_GT(dev_bdd->size, 0);
    EXPECT_EQ(dev_bdd->nVars, 8);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "Optimized device OBDD creation time: " << duration.count() << " μs" << std::endl;
    
    destroy_optimized_device_obdd(dev_bdd);
}

TEST_F(PerformanceTest, StreamManagerPerformance) {
    const int num_streams = 4;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    CudaStreamManager* manager = create_stream_manager(num_streams);
    ASSERT_NE(manager, nullptr);
    EXPECT_EQ(manager->num_streams, num_streams);
    
    // Test stream cycling
    for (int i = 0; i < 10; i++) {
        cudaStream_t stream = get_next_stream(manager);
        EXPECT_NE(stream, nullptr);
    }
    
    sync_all_streams(manager);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "Stream manager operations time: " << duration.count() << " μs" << std::endl;
    
    destroy_stream_manager(manager);
}

TEST_F(PerformanceTest, MultiGPUContextPerformance) {
    auto start = std::chrono::high_resolution_clock::now();
    
    MultiGPUContext* ctx = initialize_multi_gpu();
    ASSERT_NE(ctx, nullptr);
    EXPECT_GT(ctx->num_devices, 0);
    
    // Test device selection
    int device = select_optimal_device(ctx, 1024);
    EXPECT_GE(device, 0);
    EXPECT_LT(device, ctx->num_devices);
    
    balance_load_across_devices(ctx);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "Multi-GPU context setup time: " << duration.count() << " μs" << std::endl;
    
    destroy_multi_gpu_context(ctx);
}

TEST_F(PerformanceTest, ComplementEdgesPerformance) {
    OptimizedDeviceOBDD* dev_bdd = create_optimized_device_obdd(test_bdd);
    ASSERT_NE(dev_bdd, nullptr);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    enable_complement_edges(dev_bdd);
    
    // Test complement edge operations
    for (int i = 0; i < dev_bdd->size && i < 10; i++) {
        bool is_complement = is_complement_edge(dev_bdd, i, i+1);
        if (i % 2 == 0) {
            toggle_complement_edge(dev_bdd, i, i+1);
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "Complement edges operations time: " << duration.count() << " μs" << std::endl;
    
    destroy_optimized_device_obdd(dev_bdd);
}

TEST_F(PerformanceTest, WeakNormalizationPerformance) {
    OptimizedDeviceOBDD* dev_bdd = create_optimized_device_obdd(test_bdd);
    ASSERT_NE(dev_bdd, nullptr);
    
    WeakNormState* state = create_weak_norm_state(dev_bdd->size);
    ASSERT_NE(state, nullptr);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Force normalization by marking some nodes as dirty
    state->dirty_count = dev_bdd->size / 2;
    
    bool should_norm = should_normalize(state);
    if (should_norm) {
        perform_weak_normalization(dev_bdd, state);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "Weak normalization time: " << duration.count() << " μs" << std::endl;
    std::cout << "Normalized: " << (should_norm ? "Yes" : "No") << std::endl;
    
    destroy_weak_norm_state(state);
    destroy_optimized_device_obdd(dev_bdd);
}

TEST_F(PerformanceTest, DynamicReorderingPerformance) {
    OptimizedDeviceOBDD* dev_bdd = create_optimized_device_obdd(test_bdd);
    ASSERT_NE(dev_bdd, nullptr);
    
    DynamicReorderContext* ctx = create_reorder_context(dev_bdd->nVars);
    ASSERT_NE(ctx, nullptr);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Force reordering by setting high cost
    for (int i = 0; i < 10 && i < dev_bdd->nVars; i++) {
        ctx->level_costs[i] = 2.0; // Above threshold
    }
    
    bool should_reord = should_reorder(ctx);
    if (should_reord) {
        perform_dynamic_reordering(dev_bdd, ctx);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "Dynamic reordering time: " << duration.count() << " μs" << std::endl;
    std::cout << "Reordered: " << (should_reord ? "Yes" : "No") << std::endl;
    
    // Print final variable ordering
    std::cout << "Final variable order: ";
    for (int i = 0; i < dev_bdd->nVars; i++) {
        std::cout << ctx->current_order[i] << " ";
    }
    std::cout << std::endl;
    
    destroy_reorder_context(ctx);
    destroy_optimized_device_obdd(dev_bdd);
}

TEST_F(PerformanceTest, MemoryUsageComparison) {
    // Standard device OBDD
    size_t free_before, total_before;
    cudaMemGetInfo(&free_before, &total_before);
    size_t used_before = total_before - free_before;
    
    OptimizedDeviceOBDD* dev_bdd = create_optimized_device_obdd(test_bdd);
    ASSERT_NE(dev_bdd, nullptr);
    
    size_t free_after, total_after;
    cudaMemGetInfo(&free_after, &total_after);
    size_t used_after = total_after - free_after;
    
    size_t memory_used = used_after - used_before;
    
    std::cout << "Memory usage for optimized OBDD: " << memory_used << " bytes" << std::endl;
    std::cout << "Memory per node: " << (memory_used / dev_bdd->size) << " bytes" << std::endl;
    
    // Test complement edges memory reduction
    enable_complement_edges(dev_bdd);
    
    size_t free_optimized, total_optimized;
    cudaMemGetInfo(&free_optimized, &total_optimized);
    size_t used_optimized = total_optimized - free_optimized;
    
    std::cout << "Memory after complement edges: " << used_optimized - used_before << " bytes" << std::endl;
    
    destroy_optimized_device_obdd(dev_bdd);
}

TEST_F(PerformanceTest, OverallOptimizationBenchmark) {
    std::cout << "\n=== OVERALL OPTIMIZATION BENCHMARK ===" << std::endl;
    
    auto total_start = std::chrono::high_resolution_clock::now();
    
    // 1. Create optimized device OBDD
    OptimizedDeviceOBDD* dev_bdd = create_optimized_device_obdd(test_bdd);
    ASSERT_NE(dev_bdd, nullptr);
    
    // 2. Setup all optimization components
    CudaStreamManager* stream_mgr = create_stream_manager(4);
    MultiGPUContext* multi_gpu = initialize_multi_gpu();
    WeakNormState* weak_norm = create_weak_norm_state(dev_bdd->size);
    DynamicReorderContext* reorder_ctx = create_reorder_context(dev_bdd->nVars);
    
    // 3. Apply optimizations
    enable_complement_edges(dev_bdd);
    perform_weak_normalization(dev_bdd, weak_norm);
    
    // Force reordering
    for (int i = 0; i < dev_bdd->nVars && i < 10; i++) {
        reorder_ctx->level_costs[i] = 2.0;
    }
    perform_dynamic_reordering(dev_bdd, reorder_ctx);
    
    // 4. Sync all operations
    sync_all_streams(stream_mgr);
    balance_load_across_devices(multi_gpu);
    
    auto total_end = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start);
    
    std::cout << "Total optimization time: " << total_duration.count() << " ms" << std::endl;
    std::cout << "BDD nodes: " << dev_bdd->size << std::endl;
    std::cout << "Variables: " << dev_bdd->nVars << std::endl;
    std::cout << "GPU devices: " << multi_gpu->num_devices << std::endl;
    std::cout << "Streams: " << stream_mgr->num_streams << std::endl;
    
    // Cleanup
    destroy_reorder_context(reorder_ctx);
    destroy_weak_norm_state(weak_norm);
    destroy_multi_gpu_context(multi_gpu);
    destroy_stream_manager(stream_mgr);
    destroy_optimized_device_obdd(dev_bdd);
    
    std::cout << "=== BENCHMARK COMPLETE ===" << std::endl;
}

#else

TEST_F(PerformanceTest, CUDADisabled) {
    std::cout << "CUDA backend is disabled. Skipping performance tests." << std::endl;
    GTEST_SKIP() << "CUDA backend is disabled";
}

#endif /* OBDD_ENABLE_CUDA */