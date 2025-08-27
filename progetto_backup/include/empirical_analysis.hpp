/**
 * @file empirical_analysis.hpp
 * @brief Advanced empirical analysis tools for CUDA OBDD performance
 */

#pragma once

#ifdef OBDD_ENABLE_CUDA
#include "obdd_cuda_optimized.cuh"
#include <cuda_runtime.h>
#include <chrono>
#include <vector>
#include <string>
#include <memory>

/* =====================================================
   SCALING ANALYSIS STRUCTURES
   ===================================================== */

/**
 * @brief Strong scaling analysis results
 */
struct StrongScalingResult {
    std::vector<int> thread_counts;         // Number of threads used
    std::vector<double> execution_times;    // Execution time per thread count
    std::vector<double> speedups;           // Speedup relative to single thread
    std::vector<double> efficiencies;       // Parallel efficiency (speedup/threads)
    double problem_size;                    // Fixed problem size
    double sequential_time;                 // Single-thread baseline time
    double parallel_overhead;               // Estimated parallel overhead
};

/**
 * @brief Weak scaling analysis results
 */
struct WeakScalingResult {
    std::vector<int> thread_counts;         // Number of threads used
    std::vector<double> problem_sizes;      // Problem size per thread count
    std::vector<double> execution_times;    // Execution time per configuration
    std::vector<double> efficiencies;       // Weak scaling efficiency
    double base_time;                       // Time for base configuration
    double work_per_thread;                 // Fixed work per thread
};

/**
 * @brief Memory bandwidth analysis results
 */
struct MemoryBandwidthResult {
    double theoretical_bandwidth_gbps;      // Theoretical peak bandwidth
    double achieved_bandwidth_gbps;         // Actually achieved bandwidth
    double bandwidth_utilization;           // Percentage of theoretical peak
    size_t bytes_transferred;               // Total bytes transferred
    double transfer_time_ms;                // Time for transfers
    double read_bandwidth_gbps;             // Read-only bandwidth
    double write_bandwidth_gbps;            // Write-only bandwidth
    double effective_bandwidth_gbps;        // Effective bandwidth considering cache
};

/**
 * @brief Cache miss analysis results
 */
struct CacheMissResult {
    uint64_t l1_hits;                       // L1 cache hits
    uint64_t l1_misses;                     // L1 cache misses
    uint64_t l2_hits;                       // L2 cache hits  
    uint64_t l2_misses;                     // L2 cache misses
    uint64_t dram_accesses;                 // DRAM accesses
    double l1_hit_rate;                     // L1 cache hit percentage
    double l2_hit_rate;                     // L2 cache hit percentage
    double average_memory_latency;          // Average memory access latency
    double cache_efficiency;                // Overall cache efficiency score
};

/**
 * @brief Kernel occupancy analysis results
 */
struct KernelOccupancyResult {
    int max_threads_per_block;              // Maximum threads per block
    int max_blocks_per_sm;                  // Maximum blocks per SM
    int optimal_threads_per_block;          // Optimal configuration
    int optimal_blocks_per_sm;              // Optimal blocks per SM
    double theoretical_occupancy;           // Theoretical occupancy percentage
    double achieved_occupancy;              // Actually achieved occupancy
    int shared_memory_per_block;            // Shared memory usage
    int registers_per_thread;               // Register usage per thread
    double warp_efficiency;                 // Warp execution efficiency
};

/**
 * @brief Power consumption analysis results
 */
struct PowerConsumptionResult {
    double idle_power_watts;                // Idle power consumption
    double peak_power_watts;                // Peak power during computation
    double average_power_watts;             // Average power during test
    double energy_consumed_joules;          // Total energy consumed
    double energy_efficiency_gflops_w;      // Performance per watt
    double thermal_throttling_events;       // Number of thermal throttling events
    std::vector<double> power_timeline;     // Power consumption over time
    std::vector<double> temperature_timeline; // Temperature over time
};

/**
 * @brief Comprehensive empirical analysis context
 */
struct EmpiricalAnalysisContext {
    // Device information
    cudaDeviceProp device_properties;
    int device_id;
    
    // Analysis configurations
    std::vector<int> scaling_thread_counts;  // Thread counts for scaling analysis
    std::vector<size_t> problem_sizes;       // Problem sizes to test
    int num_iterations;                      // Number of iterations per test
    bool enable_profiling;                   // Enable CUDA profiling
    
    // Timing infrastructure
    std::vector<cudaEvent_t> cuda_events;    // CUDA timing events
    std::chrono::high_resolution_clock::time_point start_time;
    
    // Memory bandwidth test buffers
    void* d_bandwidth_src;                   // Source buffer for bandwidth tests
    void* d_bandwidth_dst;                   // Destination buffer for bandwidth tests
    size_t bandwidth_buffer_size;            // Size of bandwidth test buffers
    
    // Helper buffers for kernel parameters
    bool* d_is_normalized;                   // Normalization flags buffer
    uint32_t* d_norm_timestamps;             // Timestamps buffer
};

/* =====================================================
   FUNCTION DECLARATIONS
   ===================================================== */

#ifdef __cplusplus
extern "C" {
#endif

// Context management
EmpiricalAnalysisContext* create_empirical_analysis_context();
void destroy_empirical_analysis_context(EmpiricalAnalysisContext* ctx);
void configure_analysis_parameters(EmpiricalAnalysisContext* ctx, 
                                  const std::vector<int>& thread_counts,
                                  const std::vector<size_t>& problem_sizes,
                                  int iterations);

// Scaling analysis
StrongScalingResult* perform_strong_scaling_analysis(
    EmpiricalAnalysisContext* ctx, OptimizedDeviceOBDD* dev_bdd);
WeakScalingResult* perform_weak_scaling_analysis(
    EmpiricalAnalysisContext* ctx, const std::vector<OptimizedDeviceOBDD*>& dev_bdds);

// Memory and cache analysis
MemoryBandwidthResult* analyze_memory_bandwidth(EmpiricalAnalysisContext* ctx);
CacheMissResult* analyze_cache_performance(
    EmpiricalAnalysisContext* ctx, OptimizedDeviceOBDD* dev_bdd);

// Kernel optimization analysis  
KernelOccupancyResult* analyze_kernel_occupancy(
    EmpiricalAnalysisContext* ctx, const char* kernel_name);
KernelOccupancyResult* optimize_kernel_configuration(
    EmpiricalAnalysisContext* ctx, OptimizedDeviceOBDD* dev_bdd);

// Power consumption analysis
PowerConsumptionResult* analyze_power_consumption(
    EmpiricalAnalysisContext* ctx, OptimizedDeviceOBDD* dev_bdd, double test_duration_seconds);

// Reporting and visualization
void generate_scaling_report(const StrongScalingResult* strong, const WeakScalingResult* weak);
void generate_memory_report(const MemoryBandwidthResult* bandwidth, const CacheMissResult* cache);
void generate_optimization_report(const KernelOccupancyResult* occupancy, const PowerConsumptionResult* power);
void generate_comprehensive_report(EmpiricalAnalysisContext* ctx, const char* output_file);

// Utility functions
double calculate_gflops(size_t operations, double time_seconds);
double calculate_memory_throughput(size_t bytes, double time_seconds);
void print_device_capabilities(const cudaDeviceProp& props);

#ifdef __cplusplus
}
#endif

// C++ only functions (cannot be in extern "C" due to STL usage)
#ifdef __cplusplus
// Performance comparison functions
void compare_optimization_impact(
    const MemoryBandwidthResult* baseline_bandwidth,
    const MemoryBandwidthResult* optimized_bandwidth,
    const CacheMissResult* baseline_cache,
    const CacheMissResult* optimized_cache);

void generate_performance_summary_table(
    const StrongScalingResult* strong,
    const MemoryBandwidthResult* bandwidth,
    const KernelOccupancyResult* occupancy,
    const PowerConsumptionResult* power);
#endif

#endif /* OBDD_ENABLE_CUDA */