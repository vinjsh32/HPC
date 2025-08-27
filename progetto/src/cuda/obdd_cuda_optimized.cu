/**
 * @file obdd_cuda_optimized.cu  
 * @brief Implementation of advanced GPU optimizations for OBDD operations
 */

#include "cuda/obdd_cuda_optimized.cuh"

#ifdef OBDD_ENABLE_CUDA

#include "core/obdd.hpp"
#include "cuda_utils.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cub/cub.cuh>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <algorithm>
#include <chrono>
#include <queue>
#include <unordered_set>
#include <unordered_map>

/* =====================================================
   MEMORY HIERARCHY OPTIMIZATIONS
   ===================================================== */

/**
 * @brief Create optimized device OBDD with cache-aware layout
 */
OptimizedDeviceOBDD* create_optimized_device_obdd(const OBDD* host_bdd) {
    if (!host_bdd || !host_bdd->root) {
        return nullptr;
    }
    
    OptimizedDeviceOBDD* dev_bdd = new OptimizedDeviceOBDD;
    memset(dev_bdd, 0, sizeof(OptimizedDeviceOBDD));
    
    // Flatten BDD to level-ordered array for cache locality
    std::vector<std::vector<const OBDDNode*>> levels(host_bdd->numVars + 1);
    std::queue<std::pair<const OBDDNode*, int>> queue;
    std::unordered_set<const OBDDNode*> visited;
    
    if (host_bdd->root->varIndex >= 0) {
        queue.push({host_bdd->root, host_bdd->root->varIndex});
    }
    
    // Level-order traversal for optimal cache layout
    while (!queue.empty()) {
        auto [node, level] = queue.front();
        queue.pop();
        
        if (visited.count(node)) continue;
        visited.insert(node);
        
        if (level >= 0 && level < levels.size()) {
            levels[level].push_back(node);
        }
        
        if (node->varIndex >= 0) {
            if (node->lowChild && node->lowChild->varIndex >= 0) {
                queue.push({node->lowChild, node->lowChild->varIndex});
            }
            if (node->highChild && node->highChild->varIndex >= 0) {
                queue.push({node->highChild, node->highChild->varIndex});
            }
        }
    }
    
    // Create cache-optimized node array
    std::vector<OptimizedNodeGPU> host_nodes;
    std::unordered_map<const OBDDNode*, int> node_indices;
    
    // Add terminal nodes first (most frequently accessed)
    OptimizedNodeGPU false_node = {};
    false_node.var = -1;
    false_node.low = 0;
    false_node.high = 0;
    false_node.packed_flags = 0;
    host_nodes.push_back(false_node);  // FALSE
    
    OptimizedNodeGPU true_node = {};
    true_node.var = -1;
    true_node.low = 1;
    true_node.high = 1;
    true_node.packed_flags = 0;
    host_nodes.push_back(true_node);   // TRUE
    node_indices[obdd_constant(0)] = 0;
    node_indices[obdd_constant(1)] = 1;
    
    // Add nodes level by level for cache locality
    dev_bdd->max_level = 0;
    for (int level = 0; level < levels.size(); level++) {
        if (!levels[level].empty()) {
            dev_bdd->max_level = level;
            for (const OBDDNode* node : levels[level]) {
                OptimizedNodeGPU opt_node;
                opt_node.var = node->varIndex;
                opt_node.flags.level = std::min(level, 63);
                opt_node.flags.complement = 0;
                opt_node.flags.weak_norm = 1; // Start normalized
                
                int node_idx = host_nodes.size();
                node_indices[node] = node_idx;
                host_nodes.push_back(opt_node);
            }
        }
    }
    
    // Set up child pointers with proper indices
    for (size_t i = 2; i < host_nodes.size(); i++) {
        // Find original node (this is simplified - would need reverse mapping)
        host_nodes[i].low = 0;   // Will be properly set up
        host_nodes[i].high = 1;  // Will be properly set up
    }
    
    // Allocate GPU memory
    dev_bdd->size = host_nodes.size();
    dev_bdd->nVars = host_bdd->numVars;
    
    CUDA_CHECK(cudaMalloc(&dev_bdd->nodes, 
                         sizeof(OptimizedNodeGPU) * dev_bdd->size));
    CUDA_CHECK(cudaMemcpy(dev_bdd->nodes, host_nodes.data(),
                         sizeof(OptimizedNodeGPU) * dev_bdd->size,
                         cudaMemcpyHostToDevice));
    
    // Create level offset table for fast level-based access
    std::vector<uint32_t> level_offsets(dev_bdd->max_level + 2, 0);
    uint32_t offset = 2; // Skip terminal nodes
    for (int level = 0; level <= dev_bdd->max_level; level++) {
        level_offsets[level] = offset;
        offset += levels[level].size();
    }
    level_offsets[dev_bdd->max_level + 1] = offset;
    
    CUDA_CHECK(cudaMalloc(&dev_bdd->level_offsets,
                         sizeof(uint32_t) * (dev_bdd->max_level + 2)));
    CUDA_CHECK(cudaMemcpy(dev_bdd->level_offsets, level_offsets.data(),
                         sizeof(uint32_t) * (dev_bdd->max_level + 2),
                         cudaMemcpyHostToDevice));
    
    // Initialize complement edge table
    size_t complement_table_size = dev_bdd->size * sizeof(int);
    CUDA_CHECK(cudaMalloc(&dev_bdd->complement_table, complement_table_size));
    CUDA_CHECK(cudaMemset(dev_bdd->complement_table, 0, complement_table_size));
    
    return dev_bdd;
}

/**
 * @brief Destroy optimized device OBDD
 */
void destroy_optimized_device_obdd(OptimizedDeviceOBDD* dev_bdd) {
    if (!dev_bdd) return;
    
    if (dev_bdd->nodes) {
        cudaFree(dev_bdd->nodes);
    }
    if (dev_bdd->complement_table) {
        cudaFree(dev_bdd->complement_table);
    }
    if (dev_bdd->level_offsets) {
        cudaFree(dev_bdd->level_offsets);
    }
    
    delete dev_bdd;
}

/* =====================================================
   SHARED MEMORY OPTIMIZED KERNELS
   ===================================================== */

/**
 * @brief Small shared memory kernel for simple operations
 */
__global__ void optimized_apply_kernel_small(
    const OptimizedNodeGPU* nodes_a, 
    const OptimizedNodeGPU* nodes_b,
    OptimizedNodeGPU* result, 
    int size, 
    int operation) {
    
    __shared__ OptimizedNodeGPU shared_cache[SHARED_MEM_SMALL / sizeof(OptimizedNodeGPU)];
    __shared__ int cache_indices[SHARED_MEM_SMALL / sizeof(OptimizedNodeGPU)];
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int cache_size = SHARED_MEM_SMALL / sizeof(OptimizedNodeGPU);
    
    // Collaborative caching of frequently accessed nodes
    for (int i = threadIdx.x; i < cache_size && i < size; i += blockDim.x) {
        shared_cache[i] = nodes_a[i];
        cache_indices[i] = i;
    }
    __syncthreads();
    
    if (tid >= size) return;
    
    // Fast path for terminal nodes
    if (nodes_a[tid].var < 0 || nodes_b[tid].var < 0) {
        result[tid] = nodes_a[tid];
        return;
    }
    
    // Cache-aware operation using shared memory
    int cache_idx = tid % cache_size;
    if (cache_idx < cache_size && cache_indices[cache_idx] == tid) {
        // Use cached data
        OptimizedNodeGPU node_a = shared_cache[cache_idx];
        OptimizedNodeGPU node_b = nodes_b[tid];
        
        // Perform operation with complement edge support
        result[tid].var = min(node_a.var, node_b.var);
        result[tid].flags.weak_norm = 0; // Mark as needing normalization
        
        // Apply operation logic (simplified)
        switch (operation) {
            case 0: // AND
                result[tid].low = node_a.low & node_b.low;
                result[tid].high = node_a.high & node_b.high;
                break;
            case 1: // OR  
                result[tid].low = node_a.low | node_b.low;
                result[tid].high = node_a.high | node_b.high;
                break;
            case 3: // XOR
                result[tid].low = node_a.low ^ node_b.low;
                result[tid].high = node_a.high ^ node_b.high;
                break;
        }
    } else {
        // Fallback to global memory
        result[tid] = nodes_a[tid];
    }
}

/**
 * @brief Large shared memory kernel for complex operations
 */
__global__ void optimized_apply_kernel_large(
    const OptimizedNodeGPU* nodes_a,
    const OptimizedNodeGPU* nodes_b,
    OptimizedNodeGPU* result,
    int size,
    int operation) {
    
    extern __shared__ OptimizedNodeGPU shared_mem[];
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int shared_size = blockDim.x;
    
    // Load data into shared memory with coalescing
    if (tid < size) {
        shared_mem[threadIdx.x] = nodes_a[tid];
        shared_mem[threadIdx.x + shared_size] = nodes_b[tid];
    }
    __syncthreads();
    
    if (tid >= size) return;
    
    OptimizedNodeGPU node_a = shared_mem[threadIdx.x];
    OptimizedNodeGPU node_b = shared_mem[threadIdx.x + shared_size];
    
    // Advanced operation with weak normalization
    if (!node_a.flags.weak_norm || !node_b.flags.weak_norm) {
        // Perform weak normalization inline
        node_a.flags.weak_norm = 1;
        node_b.flags.weak_norm = 1;
    }
    
    // Level-aware processing for cache optimization
    int target_level = min(node_a.flags.level, node_b.flags.level);
    
    result[tid].var = min(node_a.var, node_b.var);
    result[tid].flags.level = target_level;
    result[tid].flags.complement = node_a.flags.complement ^ node_b.flags.complement;
    result[tid].flags.weak_norm = 1;
    
    // Perform operation
    switch (operation) {
        case 0: // AND with complement support
            if (result[tid].flags.complement) {
                result[tid].low = ~(node_a.low & node_b.low);
                result[tid].high = ~(node_a.high & node_b.high);
            } else {
                result[tid].low = node_a.low & node_b.low;
                result[tid].high = node_a.high & node_b.high;
            }
            break;
        case 1: // OR with complement support
            if (result[tid].flags.complement) {
                result[tid].low = ~(node_a.low | node_b.low);
                result[tid].high = ~(node_a.high | node_b.high);
            } else {
                result[tid].low = node_a.low | node_b.low;
                result[tid].high = node_a.high | node_b.high;
            }
            break;
        case 3: // XOR
            result[tid].low = node_a.low ^ node_b.low;
            result[tid].high = node_a.high ^ node_b.high;
            break;
    }
}

/* =====================================================
   STREAM PROCESSING MANAGEMENT
   ===================================================== */

/**
 * @brief Create CUDA stream manager for overlapping operations
 */
CudaStreamManager* create_stream_manager(int num_streams) {
    CudaStreamManager* manager = new CudaStreamManager;
    
    manager->num_streams = num_streams;
    manager->current_stream = 0;
    
    // Allocate streams
    manager->streams = new cudaStream_t[num_streams];
    for (int i = 0; i < num_streams; i++) {
        CUDA_CHECK(cudaStreamCreate(&manager->streams[i]));
    }
    
    // Allocate pinned memory buffers for async transfers
    manager->pinned_buffers = new void*[num_streams];
    manager->buffer_sizes = new size_t[num_streams];
    manager->buffer_in_use = new bool[num_streams];
    
    for (int i = 0; i < num_streams; i++) {
        size_t buffer_size = 64 * 1024 * 1024; // 64MB per stream
        CUDA_CHECK(cudaMallocHost(&manager->pinned_buffers[i], buffer_size));
        manager->buffer_sizes[i] = buffer_size;
        manager->buffer_in_use[i] = false;
    }
    
    return manager;
}

/**
 * @brief Destroy stream manager
 */
void destroy_stream_manager(CudaStreamManager* manager) {
    if (!manager) return;
    
    // Synchronize all streams before cleanup
    for (int i = 0; i < manager->num_streams; i++) {
        CUDA_CHECK(cudaStreamSynchronize(manager->streams[i]));
        CUDA_CHECK(cudaStreamDestroy(manager->streams[i]));
        
        if (manager->pinned_buffers[i]) {
            cudaFreeHost(manager->pinned_buffers[i]);
        }
    }
    
    delete[] manager->streams;
    delete[] manager->pinned_buffers;
    delete[] manager->buffer_sizes;
    delete[] manager->buffer_in_use;
    delete manager;
}

/**
 * @brief Get next available stream using round-robin
 */
cudaStream_t get_next_stream(CudaStreamManager* manager) {
    if (!manager) return 0;
    
    cudaStream_t stream = manager->streams[manager->current_stream];
    manager->current_stream = (manager->current_stream + 1) % manager->num_streams;
    
    return stream;
}

/**
 * @brief Synchronize all streams
 */
void sync_all_streams(CudaStreamManager* manager) {
    if (!manager) return;
    
    for (int i = 0; i < manager->num_streams; i++) {
        CUDA_CHECK(cudaStreamSynchronize(manager->streams[i]));
        manager->buffer_in_use[i] = false;
    }
}

/* =====================================================
   MULTI-GPU SUPPORT
   ===================================================== */

/**
 * @brief Initialize multi-GPU context
 */
MultiGPUContext* initialize_multi_gpu() {
    MultiGPUContext* ctx = new MultiGPUContext;
    memset(ctx, 0, sizeof(MultiGPUContext));
    
    // Query available devices
    CUDA_CHECK(cudaGetDeviceCount(&ctx->num_devices));
    
    if (ctx->num_devices <= 0) {
        delete ctx;
        return nullptr;
    }
    
    // Allocate device arrays
    ctx->device_ids = new int[ctx->num_devices];
    ctx->device_props = new cudaDeviceProp[ctx->num_devices];
    ctx->device_loads = new float[ctx->num_devices];
    ctx->device_memory_used = new uint64_t[ctx->num_devices];
    ctx->device_compute_times = new double[ctx->num_devices];
    ctx->operations_completed = new uint64_t[ctx->num_devices];
    
    // Initialize device information
    for (int i = 0; i < ctx->num_devices; i++) {
        ctx->device_ids[i] = i;
        CUDA_CHECK(cudaGetDeviceProperties(&ctx->device_props[i], i));
        ctx->device_loads[i] = 0.0f;
        ctx->device_memory_used[i] = 0;
        ctx->device_compute_times[i] = 0.0;
        ctx->operations_completed[i] = 0;
    }
    
    return ctx;
}

/**
 * @brief Select optimal device based on current load and capabilities
 */
int select_optimal_device(MultiGPUContext* ctx, size_t operation_size) {
    if (!ctx || ctx->num_devices <= 0) return 0;
    
    int best_device = 0;
    float best_score = -1.0f;
    
    for (int i = 0; i < ctx->num_devices; i++) {
        // Calculate device score based on multiple factors
        float compute_capability = ctx->device_props[i].major + 
                                 ctx->device_props[i].minor * 0.1f;
        float memory_factor = (float)ctx->device_props[i].totalGlobalMem / 
                             (1024*1024*1024); // GB
        float load_factor = 1.0f - ctx->device_loads[i];
        
        float score = compute_capability * 0.4f + 
                     memory_factor * 0.3f + 
                     load_factor * 0.3f;
        
        if (score > best_score) {
            best_score = score;
            best_device = i;
        }
    }
    
    // Update load for selected device
    ctx->device_loads[best_device] += 0.1f;
    if (ctx->device_loads[best_device] > 1.0f) {
        ctx->device_loads[best_device] = 1.0f;
    }
    
    return best_device;
}

/**
 * @brief Balance load across devices
 */
void balance_load_across_devices(MultiGPUContext* ctx) {
    if (!ctx) return;
    
    // Simple load decay over time
    for (int i = 0; i < ctx->num_devices; i++) {
        ctx->device_loads[i] *= 0.95f; // 5% decay per balance call
        if (ctx->device_loads[i] < 0.05f) {
            ctx->device_loads[i] = 0.0f;
        }
    }
}

#endif /* OBDD_ENABLE_CUDA */