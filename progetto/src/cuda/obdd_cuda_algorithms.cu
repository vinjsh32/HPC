/**
 * @file obdd_cuda_algorithms.cu
 * @brief Advanced algorithmic optimizations for CUDA OBDD operations
 */

#include "cuda/obdd_cuda_optimized.cuh"

#ifdef OBDD_ENABLE_CUDA

#include "cuda_utils.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cub/cub.cuh>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <vector>
#include <algorithm>

/* =====================================================
   FORWARD DECLARATIONS
   ===================================================== */

__global__ void identify_complement_edges_kernel(
    OptimizedNodeGPU* nodes, int* complement_table, int size);
__global__ void toggle_complement_kernel(
    OptimizedNodeGPU* nodes, int* complement_table, int node_id);
__global__ void weak_normalization_kernel(
    OptimizedNodeGPU* nodes, bool* is_normalized, 
    uint32_t* norm_timestamps, int size);
__global__ void update_variable_activities_kernel(
    const OptimizedNodeGPU* nodes, int size, 
    float* variable_activity, int num_vars);
__device__ bool are_complement_nodes(
    const OptimizedNodeGPU& node_a, const OptimizedNodeGPU& node_b);

/* =====================================================
   COMPLEMENT EDGES IMPLEMENTATION
   ===================================================== */

/**
 * @brief Enable complement edges optimization
 */
void enable_complement_edges(OptimizedDeviceOBDD* dev_bdd) {
    if (!dev_bdd || !dev_bdd->complement_table) return;
    
    // Initialize complement table - all edges start as non-complement
    CUDA_CHECK(cudaMemset(dev_bdd->complement_table, 0, 
                         dev_bdd->size * sizeof(int)));
    
    // Launch kernel to identify complement opportunities
    int threads_per_block = 256;
    int blocks = (dev_bdd->size + threads_per_block - 1) / threads_per_block;
    
    identify_complement_edges_kernel<<<blocks, threads_per_block>>>(
        dev_bdd->nodes, dev_bdd->complement_table, dev_bdd->size);
    
    CUDA_CHECK(cudaDeviceSynchronize());
}

/**
 * @brief Device function to check if two nodes are complements
 */
__device__ bool are_complement_nodes(
    const OptimizedNodeGPU& node_a, 
    const OptimizedNodeGPU& node_b) {
    
    // Terminal nodes check
    if (node_a.var < 0 && node_b.var < 0) {
        return (node_a.low != node_b.low) || (node_a.high != node_b.high);
    }
    
    // Same variable, different children patterns indicate complements
    if (node_a.var == node_b.var) {
        return (node_a.low == node_b.high) && (node_a.high == node_b.low);
    }
    
    return false;
}

/**
 * @brief Kernel to identify complement edge opportunities
 */
__global__ void identify_complement_edges_kernel(
    OptimizedNodeGPU* nodes,
    int* complement_table,
    int size) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= size) return;
    
    OptimizedNodeGPU& node = nodes[tid];
    
    // Skip terminal nodes
    if (node.var < 0) return;
    
    // Check if this node can benefit from complement edges
    // Simple heuristic: if low and high children are "opposite"
    if (node.low < size && node.high < size) {
        OptimizedNodeGPU& low_child = nodes[node.low];
        OptimizedNodeGPU& high_child = nodes[node.high];
        
        // If children are complements of each other, mark edge
        if (are_complement_nodes(low_child, high_child)) {
            complement_table[tid] = 1;
            node.flags.complement = 1;
        }
    }
}

/**
 * @brief Check if edge is a complement edge
 */
bool is_complement_edge(OptimizedDeviceOBDD* dev_bdd, int node_id, int child_id) {
    if (!dev_bdd || node_id >= dev_bdd->size) return false;
    
    int complement_flag = 0;
    CUDA_CHECK(cudaMemcpy(&complement_flag, 
                         &dev_bdd->complement_table[node_id],
                         sizeof(int), cudaMemcpyDeviceToHost));
    
    return complement_flag != 0;
}

/**
 * @brief Toggle complement edge
 */
void toggle_complement_edge(OptimizedDeviceOBDD* dev_bdd, int node_id, int child_id) {
    if (!dev_bdd || node_id >= dev_bdd->size) return;
    
    // Launch single-thread kernel to toggle
    toggle_complement_kernel<<<1, 1>>>(
        dev_bdd->nodes, dev_bdd->complement_table, node_id);
    
    CUDA_CHECK(cudaDeviceSynchronize());
}

/**
 * @brief Kernel to toggle complement edge
 */
__global__ void toggle_complement_kernel(
    OptimizedNodeGPU* nodes,
    int* complement_table,
    int node_id) {
    
    complement_table[node_id] = !complement_table[node_id];
    nodes[node_id].flags.complement = complement_table[node_id];
}

/* =====================================================
   WEAK NORMALIZATION IMPLEMENTATION
   ===================================================== */

/**
 * @brief Create weak normalization state
 */
WeakNormState* create_weak_norm_state(int num_nodes) {
    WeakNormState* state = new WeakNormState;
    
    // Allocate device memory for normalization tracking
    CUDA_CHECK(cudaMalloc(&state->is_normalized, num_nodes * sizeof(bool)));
    CUDA_CHECK(cudaMalloc(&state->norm_timestamps, num_nodes * sizeof(uint32_t)));
    
    // Initialize all nodes as normalized
    CUDA_CHECK(cudaMemset(state->is_normalized, 1, num_nodes * sizeof(bool)));
    CUDA_CHECK(cudaMemset(state->norm_timestamps, 0, num_nodes * sizeof(uint32_t)));
    
    state->dirty_count = 0;
    state->norm_threshold = 0.1f;     // Normalize when 10% nodes are dirty
    state->max_dirty_nodes = num_nodes / 10; // Max 10% dirty nodes
    
    return state;
}

/**
 * @brief Destroy weak normalization state
 */
void destroy_weak_norm_state(WeakNormState* state) {
    if (!state) return;
    
    if (state->is_normalized) {
        cudaFree(state->is_normalized);
    }
    if (state->norm_timestamps) {
        cudaFree(state->norm_timestamps);
    }
    
    delete state;
}

/**
 * @brief Check if normalization should be performed
 */
bool should_normalize(WeakNormState* state) {
    if (!state) return false;
    
    float dirty_ratio = (float)state->dirty_count / state->max_dirty_nodes;
    return dirty_ratio >= state->norm_threshold;
}

/**
 * @brief Perform weak normalization
 */
void perform_weak_normalization(OptimizedDeviceOBDD* dev_bdd, WeakNormState* state) {
    if (!dev_bdd || !state) return;
    
    int threads_per_block = 256;
    int blocks = (dev_bdd->size + threads_per_block - 1) / threads_per_block;
    
    // Launch normalization kernel
    weak_normalization_kernel<<<blocks, threads_per_block>>>(
        dev_bdd->nodes, state->is_normalized, state->norm_timestamps,
        dev_bdd->size);
    
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Reset dirty count
    state->dirty_count = 0;
}

/**
 * @brief Weak normalization kernel
 */
__global__ void weak_normalization_kernel(
    OptimizedNodeGPU* nodes,
    bool* is_normalized,
    uint32_t* norm_timestamps,
    int size) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= size) return;
    
    OptimizedNodeGPU& node = nodes[tid];
    
    // Skip already normalized nodes
    if (is_normalized[tid] && node.flags.weak_norm) return;
    
    // Skip terminal nodes
    if (node.var < 0) {
        is_normalized[tid] = true;
        node.flags.weak_norm = 1;
        return;
    }
    
    // Perform weak normalization
    // Ensure low child has lower or equal variable index
    if (node.low < size && node.high < size) {
        OptimizedNodeGPU& low_child = nodes[node.low];
        OptimizedNodeGPU& high_child = nodes[node.high];
        
        // Swap if needed for canonical ordering
        if (low_child.var > high_child.var) {
            int temp = node.low;
            node.low = node.high;
            node.high = temp;
        }
    }
    
    // Mark as normalized
    is_normalized[tid] = true;
    node.flags.weak_norm = 1;
    norm_timestamps[tid] = clock(); // Simple timestamp
}

/* =====================================================
   DYNAMIC VARIABLE REORDERING IMPLEMENTATION
   ===================================================== */

/**
 * @brief Create dynamic reordering context
 */
DynamicReorderContext* create_reorder_context(int num_vars) {
    DynamicReorderContext* ctx = new DynamicReorderContext;
    
    // Allocate arrays
    ctx->current_order = new int[num_vars];
    ctx->optimal_order = new int[num_vars];
    ctx->variable_activity = new float[num_vars];
    ctx->swap_counts = new uint64_t[num_vars * num_vars];
    ctx->level_costs = new double[num_vars];
    
    // Initialize with identity ordering
    for (int i = 0; i < num_vars; i++) {
        ctx->current_order[i] = i;
        ctx->optimal_order[i] = i;
        ctx->variable_activity[i] = 1.0f;
        ctx->level_costs[i] = 1.0;
    }
    
    // Initialize swap counts matrix
    memset(ctx->swap_counts, 0, num_vars * num_vars * sizeof(uint64_t));
    
    // Set parameters
    ctx->reorder_threshold = 1.5f;    // Reorder when cost increases 50%
    ctx->reorder_frequency = 1000;    // Check every 1000 operations
    ctx->adaptive_enabled = true;
    
    return ctx;
}

/**
 * @brief Destroy reordering context
 */
void destroy_reorder_context(DynamicReorderContext* ctx) {
    if (!ctx) return;
    
    delete[] ctx->current_order;
    delete[] ctx->optimal_order;
    delete[] ctx->variable_activity;
    delete[] ctx->swap_counts;
    delete[] ctx->level_costs;
    delete ctx;
}

/**
 * @brief Check if reordering should be performed
 */
bool should_reorder(DynamicReorderContext* ctx) {
    if (!ctx || !ctx->adaptive_enabled) return false;
    
    // Simple heuristic: check if average level cost exceeds threshold
    double avg_cost = 0.0;
    for (int i = 0; ctx->level_costs && i < 10; i++) { // Check first 10 levels
        avg_cost += ctx->level_costs[i];
    }
    avg_cost /= 10.0;
    
    return avg_cost > ctx->reorder_threshold;
}

/**
 * @brief Perform dynamic variable reordering
 */
void perform_dynamic_reordering(OptimizedDeviceOBDD* dev_bdd, 
                               DynamicReorderContext* ctx) {
    if (!dev_bdd || !ctx) return;
    
    // Update variable activities based on current BDD structure
    update_variable_activities_kernel<<<
        (dev_bdd->size + 255) / 256, 256>>>(
        dev_bdd->nodes, dev_bdd->size, ctx->variable_activity, dev_bdd->nVars);
    
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Compute optimal ordering using simple heuristic
    // (In practice, would use more sophisticated algorithms like sifting)
    std::vector<std::pair<float, int>> activity_pairs;
    for (int i = 0; i < dev_bdd->nVars; i++) {
        activity_pairs.push_back({ctx->variable_activity[i], i});
    }
    
    // Sort by activity (most active variables first)
    std::sort(activity_pairs.begin(), activity_pairs.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });
    
    // Update optimal ordering
    for (int i = 0; i < dev_bdd->nVars; i++) {
        ctx->optimal_order[i] = activity_pairs[i].second;
    }
    
    // Apply reordering (simplified - would need full BDD reconstruction)
    memcpy(ctx->current_order, ctx->optimal_order, 
           dev_bdd->nVars * sizeof(int));
}

/**
 * @brief Kernel to update variable activities
 */
__global__ void update_variable_activities_kernel(
    const OptimizedNodeGPU* nodes,
    int size,
    float* variable_activity,
    int num_vars) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= size) return;
    
    const OptimizedNodeGPU& node = nodes[tid];
    
    // Skip terminal nodes
    if (node.var < 0 || node.var >= num_vars) return;
    
    // Increment activity for this variable (atomic for thread safety)
    atomicAdd(&variable_activity[node.var], 1.0f);
    
    // Boost activity if node is frequently accessed (based on level)
    if (node.flags.level < 5) { // Boost top 5 levels
        atomicAdd(&variable_activity[node.var], 0.5f);
    }
}

/* =====================================================
   PERFORMANCE MONITORING KERNELS
   ===================================================== */

/**
 * @brief Kernel to collect cache performance statistics
 */
__global__ void collect_cache_stats_kernel(
    OptimizedDeviceOBDD* dev_bdd,
    uint64_t* cache_hits,
    uint64_t* cache_misses) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Simple cache simulation
    __shared__ int cache_simulation[256];
    
    if (threadIdx.x < 256) {
        cache_simulation[threadIdx.x] = -1;
    }
    __syncthreads();
    
    if (tid >= dev_bdd->size) return;
    
    int cache_slot = tid % 256;
    
    if (cache_simulation[cache_slot] == tid) {
        atomicAdd((unsigned long long*)cache_hits, 1);
    } else {
        atomicAdd((unsigned long long*)cache_misses, 1);
        cache_simulation[cache_slot] = tid;
    }
}

#endif /* OBDD_ENABLE_CUDA */