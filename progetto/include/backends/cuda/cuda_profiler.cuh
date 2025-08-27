/**
 * @file cuda_profiler.cuh
 * @brief Advanced CUDA profiling system for detailed GPU performance analysis
 * @version 2.0
 * @date 2024
 * 
 * This module provides comprehensive CUDA profiling capabilities including:
 * - High-precision GPU timing with CUDA Events
 * - Memory bandwidth measurement and analysis
 * - Kernel occupancy and warp efficiency metrics
 * - Memory coalescing analysis
 * - GPU utilization and power consumption tracking
 * 
 * @author HPC Team
 * @copyright 2024 High Performance Computing Laboratory
 */

#pragma once

#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <device_launch_parameters.h>
#include <cuda_occupancy.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Comprehensive GPU profiling metrics
 */
typedef struct {
    // Timing metrics (high precision)
    float kernel_time_ms;
    float memory_transfer_time_ms;
    float total_gpu_time_ms;
    
    // Memory bandwidth metrics
    float effective_bandwidth_gbps;
    float theoretical_bandwidth_gbps;
    float bandwidth_efficiency_percent;
    
    // Kernel execution metrics
    int blocks_per_sm;
    int warps_per_sm;
    int threads_per_warp;
    float occupancy_percent;
    
    // Memory access patterns
    int coalesced_transactions;
    int uncoalesced_transactions;
    float coalescing_efficiency_percent;
    
    // Resource utilization
    float sm_utilization_percent;
    float memory_utilization_percent;
    float instruction_throughput_percent;
    
    // Power and thermal
    float power_consumption_watts;
    float gpu_temperature_celsius;
    
    // Cache statistics
    float l1_cache_hit_rate_percent;
    float l2_cache_hit_rate_percent;
    
    // Error tracking
    int cuda_errors_count;
    char last_error_message[256];
    
} CudaProfilingMetrics;

/**
 * @brief CUDA profiler context for session management
 */
typedef struct {
    cudaEvent_t start_event;
    cudaEvent_t stop_event;
    cudaEvent_t kernel_start;
    cudaEvent_t kernel_stop;
    
    // Memory tracking
    size_t initial_free_memory;
    size_t peak_memory_usage;
    size_t bytes_transferred_h2d;
    size_t bytes_transferred_d2h;
    
    // Timing accumulation
    float total_kernel_time;
    float total_memory_time;
    int kernel_launches;
    
    // Device properties cache
    cudaDeviceProp device_props;
    int device_id;
    
    bool profiling_active;
    
} CudaProfilerContext;

// =====================================================
// Profiler Lifecycle Management
// =====================================================

/**
 * @brief Initialize CUDA profiler context
 * @param ctx Profiler context to initialize
 * @param device_id CUDA device ID (use -1 for current device)
 * @return cudaSuccess on success, error code on failure
 */
cudaError_t cuda_profiler_init(CudaProfilerContext* ctx, int device_id);

/**
 * @brief Start profiling session
 * @param ctx Profiler context
 * @return cudaSuccess on success, error code on failure
 */
cudaError_t cuda_profiler_start(CudaProfilerContext* ctx);

/**
 * @brief Stop profiling session and collect metrics
 * @param ctx Profiler context
 * @param metrics Output structure for collected metrics
 * @return cudaSuccess on success, error code on failure
 */
cudaError_t cuda_profiler_stop(CudaProfilerContext* ctx, CudaProfilingMetrics* metrics);

/**
 * @brief Cleanup profiler context and free resources
 * @param ctx Profiler context to cleanup
 */
void cuda_profiler_cleanup(CudaProfilerContext* ctx);

// =====================================================
// Kernel-Level Profiling
// =====================================================

/**
 * @brief Start timing for a specific kernel
 * @param ctx Profiler context
 * @param kernel_name Name of kernel for tracking
 * @return cudaSuccess on success, error code on failure
 */
cudaError_t cuda_profiler_kernel_start(CudaProfilerContext* ctx, const char* kernel_name);

/**
 * @brief Stop timing for current kernel
 * @param ctx Profiler context
 * @return Kernel execution time in milliseconds
 */
float cuda_profiler_kernel_stop(CudaProfilerContext* ctx);

/**
 * @brief Measure kernel occupancy for given parameters
 * @param kernel_func Kernel function pointer
 * @param block_size Block size to analyze
 * @param dynamic_smem Dynamic shared memory per block
 * @param occupancy Output: occupancy percentage
 * @param blocks_per_sm Output: blocks per SM
 * @return cudaSuccess on success, error code on failure
 */
cudaError_t cuda_profiler_measure_occupancy(const void* kernel_func, int block_size, 
                                           size_t dynamic_smem, float* occupancy, 
                                           int* blocks_per_sm);

// =====================================================
// Memory Analysis
// =====================================================

/**
 * @brief Analyze memory bandwidth for data transfer
 * @param bytes_transferred Number of bytes transferred
 * @param transfer_time_ms Transfer time in milliseconds
 * @param bandwidth_gbps Output: achieved bandwidth in GB/s
 * @param efficiency_percent Output: efficiency vs theoretical peak
 * @return cudaSuccess on success, error code on failure
 */
cudaError_t cuda_profiler_analyze_bandwidth(size_t bytes_transferred, float transfer_time_ms,
                                          float* bandwidth_gbps, float* efficiency_percent);

/**
 * @brief Monitor GPU memory usage during operation
 * @param ctx Profiler context
 * @param current_free Output: current free memory in bytes
 * @param current_total Output: total GPU memory in bytes
 * @return cudaSuccess on success, error code on failure
 */
cudaError_t cuda_profiler_monitor_memory(CudaProfilerContext* ctx, size_t* current_free, 
                                        size_t* current_total);

/**
 * @brief Analyze memory access patterns for coalescing
 * @param access_pattern Array of memory addresses accessed
 * @param num_accesses Number of memory accesses
 * @param coalescing_efficiency Output: coalescing efficiency (0.0-1.0)
 * @return cudaSuccess on success, error code on failure
 */
cudaError_t cuda_profiler_analyze_coalescing(const void** access_pattern, int num_accesses,
                                           float* coalescing_efficiency);

// =====================================================
// Advanced Metrics Collection
// =====================================================

/**
 * @brief Collect detailed SM utilization metrics
 * @param ctx Profiler context
 * @param sm_utilization Output: SM utilization percentage
 * @param instruction_throughput Output: instruction throughput percentage
 * @return cudaSuccess on success, error code on failure
 */
cudaError_t cuda_profiler_collect_sm_metrics(CudaProfilerContext* ctx, float* sm_utilization,
                                            float* instruction_throughput);

/**
 * @brief Measure cache hit rates (requires CUPTI)
 * @param ctx Profiler context  
 * @param l1_hit_rate Output: L1 cache hit rate percentage
 * @param l2_hit_rate Output: L2 cache hit rate percentage
 * @return cudaSuccess on success, error code on failure
 */
cudaError_t cuda_profiler_measure_cache_performance(CudaProfilerContext* ctx, 
                                                   float* l1_hit_rate, float* l2_hit_rate);

/**
 * @brief Get GPU power consumption and temperature
 * @param ctx Profiler context
 * @param power_watts Output: power consumption in watts
 * @param temperature_celsius Output: GPU temperature in Celsius
 * @return cudaSuccess on success, error code on failure
 */
cudaError_t cuda_profiler_get_power_thermal(CudaProfilerContext* ctx, float* power_watts,
                                          float* temperature_celsius);

// =====================================================
// Performance Analysis Functions
// =====================================================

/**
 * @brief Calculate theoretical peak performance for current GPU
 * @param ctx Profiler context
 * @param peak_gflops Output: theoretical peak GFLOPS
 * @param peak_bandwidth_gbps Output: theoretical peak memory bandwidth GB/s
 * @return cudaSuccess on success, error code on failure
 */
cudaError_t cuda_profiler_get_theoretical_peak(CudaProfilerContext* ctx, float* peak_gflops,
                                              float* peak_bandwidth_gbps);

/**
 * @brief Analyze kernel performance bottlenecks
 * @param metrics Collected profiling metrics
 * @param bottleneck_analysis Output buffer for analysis text
 * @param buffer_size Size of analysis buffer
 * @return Number of bottlenecks identified
 */
int cuda_profiler_analyze_bottlenecks(const CudaProfilingMetrics* metrics,
                                     char* bottleneck_analysis, size_t buffer_size);

/**
 * @brief Generate optimization recommendations
 * @param metrics Collected profiling metrics
 * @param recommendations Output buffer for recommendations
 * @param buffer_size Size of recommendations buffer
 * @return Number of recommendations generated
 */
int cuda_profiler_generate_recommendations(const CudaProfilingMetrics* metrics,
                                          char* recommendations, size_t buffer_size);

// =====================================================
// Reporting and Visualization
// =====================================================

/**
 * @brief Print detailed profiling report to console
 * @param metrics Collected profiling metrics
 * @param device_name GPU device name for context
 */
void cuda_profiler_print_report(const CudaProfilingMetrics* metrics, const char* device_name);

/**
 * @brief Export profiling data to JSON format
 * @param metrics Array of profiling metrics
 * @param num_metrics Number of metric entries
 * @param output_file Output JSON file path
 * @return 0 on success, -1 on failure
 */
int cuda_profiler_export_json(const CudaProfilingMetrics* metrics, int num_metrics,
                             const char* output_file);

/**
 * @brief Generate performance summary statistics
 * @param metrics Array of profiling metrics from multiple runs
 * @param num_runs Number of benchmark runs
 * @param summary_stats Output structure for summary statistics
 */
void cuda_profiler_calculate_summary(const CudaProfilingMetrics* metrics, int num_runs,
                                    CudaProfilingMetrics* summary_stats);

// =====================================================
// Utility Functions
// =====================================================

/**
 * @brief Check if advanced profiling features are available
 * @return 1 if available, 0 if not available
 */
int cuda_profiler_is_advanced_available(void);

/**
 * @brief Get readable error message for CUDA error codes
 * @param error CUDA error code
 * @return Human-readable error description
 */
const char* cuda_profiler_get_error_string(cudaError_t error);

/**
 * @brief Validate profiling metrics for correctness
 * @param metrics Metrics to validate
 * @return 1 if valid, 0 if invalid
 */
int cuda_profiler_validate_metrics(const CudaProfilingMetrics* metrics);

#ifdef __cplusplus
}
#endif

#endif // __CUDACC__