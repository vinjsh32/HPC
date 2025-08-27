#pragma once
#ifdef OBDD_ENABLE_CUDA

/**
 * @file obdd_cuda_optimized.cuh
 * @brief Advanced GPU optimizations for OBDD operations
 * 
 * Features:
 * - Cache-aware data layouts for optimal memory access patterns
 * - Shared memory optimizations for frequent operations
 * - Stream processing for overlapping computation/transfer
 * - Multi-GPU support with intelligent load balancing
 * - Complement edges to reduce memory footprint
 * - Weak normalization for faster operations
 * - Dynamic variable reordering during computation
 */

#include "obdd_cuda_types.cuh"
#include "obdd.hpp"
#include <cuda_runtime.h>
#include <vector>
#include <memory>

/* =====================================================
   CACHE-AWARE DATA STRUCTURES
   ===================================================== */

/**
 * @brief Cache-optimized node structure with improved memory layout
 * Uses 128-bit alignment and packs frequently accessed fields together
 */
struct __align__(16) OptimizedNodeGPU {
    int var;        // Variable index (-1 for leaves)
    int low;        // Low child index
    int high;       // High child index  
    union {
        struct {
            uint8_t complement : 1;  // Complement edge flag
            uint8_t weak_norm : 1;   // Weak normalization flag
            uint8_t level : 6;       // BDD level for cache locality
        } flags;
        uint8_t packed_flags;
    };
    
    // Cache line padding for optimal access patterns
    uint8_t padding[12];
};

/**
 * @brief Memory-efficient DeviceOBDD with complement edges support
 */
struct OptimizedDeviceOBDD {
    OptimizedNodeGPU* nodes;     // Device pointer to nodes
    int* complement_table;       // Complement edge lookup table
    uint32_t* level_offsets;     // Level-based access optimization
    
    int size;                    // Total number of nodes
    int nVars;                   // Number of variables
    int max_level;               // Maximum BDD level
    
    // Performance counters
    uint64_t cache_hits;
    uint64_t cache_misses;
};

/**
 * @brief Stream management for overlapping operations
 */
struct CudaStreamManager {
    cudaStream_t* streams;
    int num_streams;
    int current_stream;
    
    // Async memory pools
    void** pinned_buffers;
    size_t* buffer_sizes;
    bool* buffer_in_use;
};

/**
 * @brief Multi-GPU context and load balancer
 */
struct MultiGPUContext {
    int num_devices;
    int* device_ids;
    cudaDeviceProp* device_props;
    
    // Load balancing
    float* device_loads;         // Current load per device (0.0 - 1.0)
    uint64_t* device_memory_used; // Memory usage per device
    
    // Performance metrics
    double* device_compute_times;
    uint64_t* operations_completed;
};

/* =====================================================
   SHARED MEMORY OPTIMIZATIONS  
   ===================================================== */

/**
 * @brief Shared memory configuration for different kernel types
 */
enum SharedMemConfig {
    SHARED_MEM_SMALL = 1024,     // For simple operations
    SHARED_MEM_MEDIUM = 4096,    // For moderate complexity
    SHARED_MEM_LARGE = 8192,     // For complex multi-level operations
    SHARED_MEM_XLARGE = 16384    // For massive operations
};

/**
 * @brief Cache-aware operation parameters
 */
struct KernelConfig {
    dim3 grid_size;
    dim3 block_size;
    size_t shared_mem_size;
    cudaStream_t stream;
    
    // Cache optimization hints
    bool prefer_l1_cache;
    bool use_texture_cache;
    int memory_access_pattern;   // 0=random, 1=sequential, 2=strided
};

/* =====================================================
   WEAK NORMALIZATION STRUCTURES
   ===================================================== */

/**
 * @brief Weak normalization state for faster operations
 */
struct WeakNormState {
    bool* is_normalized;         // Per-node normalization status
    uint32_t* norm_timestamps;   // Last normalization time
    int dirty_count;             // Number of non-normalized nodes
    
    // Normalization thresholds
    float norm_threshold;        // When to trigger normalization
    int max_dirty_nodes;        // Maximum dirty nodes before forced norm
};

/* =====================================================
   DYNAMIC REORDERING STRUCTURES  
   ===================================================== */

/**
 * @brief Dynamic variable reordering context
 */
struct DynamicReorderContext {
    int* current_order;          // Current variable ordering
    int* optimal_order;          // Computed optimal ordering
    float* variable_activity;    // Variable activity scores
    
    // Reordering statistics
    uint64_t* swap_counts;       // Number of swaps per variable pair
    double* level_costs;         // Cost per BDD level
    
    // Adaptive parameters
    float reorder_threshold;     // When to trigger reordering
    int reorder_frequency;       // How often to check for reordering
    bool adaptive_enabled;       // Enable adaptive reordering
};

/* =====================================================
   FUNCTION DECLARATIONS
   ===================================================== */

#ifdef __cplusplus
extern "C" {
#endif

// Memory hierarchy optimizations
OptimizedDeviceOBDD* create_optimized_device_obdd(const OBDD* host_bdd);
void destroy_optimized_device_obdd(OptimizedDeviceOBDD* dev_bdd);
void optimize_memory_layout(OptimizedDeviceOBDD* dev_bdd);

// Shared memory kernel optimizations  
__global__ void optimized_apply_kernel_small(
    const OptimizedNodeGPU* nodes_a, const OptimizedNodeGPU* nodes_b,
    OptimizedNodeGPU* result, int size, int operation);

__global__ void optimized_apply_kernel_large(
    const OptimizedNodeGPU* nodes_a, const OptimizedNodeGPU* nodes_b, 
    OptimizedNodeGPU* result, int size, int operation);

// Stream processing
CudaStreamManager* create_stream_manager(int num_streams);
void destroy_stream_manager(CudaStreamManager* manager);
cudaStream_t get_next_stream(CudaStreamManager* manager);
void sync_all_streams(CudaStreamManager* manager);

// Multi-GPU support
MultiGPUContext* initialize_multi_gpu();
void destroy_multi_gpu_context(MultiGPUContext* ctx);
int select_optimal_device(MultiGPUContext* ctx, size_t operation_size);
void balance_load_across_devices(MultiGPUContext* ctx);

// Complement edges
void enable_complement_edges(OptimizedDeviceOBDD* dev_bdd);
bool is_complement_edge(OptimizedDeviceOBDD* dev_bdd, int node_id, int child_id);
void toggle_complement_edge(OptimizedDeviceOBDD* dev_bdd, int node_id, int child_id);

// Weak normalization
WeakNormState* create_weak_norm_state(int num_nodes);
void destroy_weak_norm_state(WeakNormState* state);
bool should_normalize(WeakNormState* state);
void perform_weak_normalization(OptimizedDeviceOBDD* dev_bdd, WeakNormState* state);

// Dynamic variable reordering
DynamicReorderContext* create_reorder_context(int num_vars);
void destroy_reorder_context(DynamicReorderContext* ctx);
bool should_reorder(DynamicReorderContext* ctx);
void perform_dynamic_reordering(OptimizedDeviceOBDD* dev_bdd, DynamicReorderContext* ctx);

#ifdef __cplusplus
}
#endif

#endif /* OBDD_ENABLE_CUDA */