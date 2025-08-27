/**
 * @file empirical_reporting.cu
 * @brief Report generation for empirical analysis results
 */

#include "cuda/empirical_analysis.hpp"

#ifdef OBDD_ENABLE_CUDA

#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <cmath>

/* =====================================================
   REPORT GENERATION FUNCTIONS
   ===================================================== */

void generate_scaling_report(const StrongScalingResult* strong, const WeakScalingResult* weak) {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "              SCALING ANALYSIS REPORT" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    
    if (strong) {
        std::cout << "\n--- STRONG SCALING ANALYSIS ---" << std::endl;
        std::cout << "Problem size: " << (int)strong->problem_size << " nodes" << std::endl;
        std::cout << "Sequential baseline: " << std::fixed << std::setprecision(3) 
                  << strong->sequential_time << " ms" << std::endl;
        
        std::cout << "\nThreads\tTime(ms)\tSpeedup\t\tEfficiency" << std::endl;
        std::cout << std::string(50, '-') << std::endl;
        
        for (size_t i = 0; i < strong->thread_counts.size(); i++) {
            std::cout << strong->thread_counts[i] << "\t"
                      << std::setprecision(3) << strong->execution_times[i] << "\t\t"
                      << std::setprecision(2) << strong->speedups[i] << "x\t\t"
                      << std::setprecision(1) << (strong->efficiencies[i] * 100) << "%" << std::endl;
        }
        
        // Find optimal configuration
        auto max_speedup_it = std::max_element(strong->speedups.begin(), strong->speedups.end());
        int optimal_idx = std::distance(strong->speedups.begin(), max_speedup_it);
        
        std::cout << "\nOptimal configuration: " << strong->thread_counts[optimal_idx] 
                  << " threads (speedup: " << std::setprecision(2) << *max_speedup_it << "x)" << std::endl;
        
        // Calculate scalability metrics
        double amdahl_overhead = 0.0;
        if (!strong->speedups.empty()) {
            int max_threads = *std::max_element(strong->thread_counts.begin(), strong->thread_counts.end());
            double max_speedup = *std::max_element(strong->speedups.begin(), strong->speedups.end());
            amdahl_overhead = (1.0 / max_speedup - 1.0 / max_threads) / (1.0 - 1.0 / max_threads);
        }
        std::cout << "Estimated serial fraction (Amdahl): " << std::setprecision(3) << amdahl_overhead << std::endl;
    }
    
    if (weak) {
        std::cout << "\n--- WEAK SCALING ANALYSIS ---" << std::endl;
        std::cout << "Work per thread: " << std::fixed << std::setprecision(1) << weak->work_per_thread << " nodes" << std::endl;
        std::cout << "Base time: " << std::setprecision(3) << weak->base_time << " ms" << std::endl;
        
        std::cout << "\nThreads\tProblem Size\tTime(ms)\tEfficiency" << std::endl;
        std::cout << std::string(50, '-') << std::endl;
        
        for (size_t i = 0; i < weak->thread_counts.size(); i++) {
            std::cout << weak->thread_counts[i] << "\t"
                      << (int)weak->problem_sizes[i] << "\t\t"
                      << std::setprecision(3) << weak->execution_times[i] << "\t\t"
                      << std::setprecision(1) << (weak->efficiencies[i] * 100) << "%" << std::endl;
        }
        
        // Calculate average efficiency
        if (!weak->efficiencies.empty()) {
            double avg_efficiency = 0.0;
            for (double eff : weak->efficiencies) {
                avg_efficiency += eff;
            }
            avg_efficiency /= weak->efficiencies.size();
            std::cout << "\nAverage weak scaling efficiency: " << std::setprecision(1) 
                      << (avg_efficiency * 100) << "%" << std::endl;
        }
    }
    
    std::cout << std::string(60, '=') << std::endl;
}

void generate_memory_report(const MemoryBandwidthResult* bandwidth, const CacheMissResult* cache) {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "            MEMORY PERFORMANCE REPORT" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    
    if (bandwidth) {
        std::cout << "\n--- MEMORY BANDWIDTH ANALYSIS ---" << std::endl;
        std::cout << "Theoretical peak bandwidth: " << std::fixed << std::setprecision(1) 
                  << bandwidth->theoretical_bandwidth_gbps << " GB/s" << std::endl;
        std::cout << "Achieved bandwidth: " << bandwidth->achieved_bandwidth_gbps << " GB/s" << std::endl;
        std::cout << "Bandwidth utilization: " << std::setprecision(1) 
                  << (bandwidth->bandwidth_utilization * 100) << "%" << std::endl;
        std::cout << "Transfer time: " << std::setprecision(3) << bandwidth->transfer_time_ms << " ms" << std::endl;
        std::cout << "Data transferred: " << (bandwidth->bytes_transferred / (1024*1024)) << " MB" << std::endl;
        
        // Performance classification
        std::string performance_class;
        if (bandwidth->bandwidth_utilization > 0.8) {
            performance_class = "Excellent";
        } else if (bandwidth->bandwidth_utilization > 0.6) {
            performance_class = "Good";
        } else if (bandwidth->bandwidth_utilization > 0.4) {
            performance_class = "Moderate";
        } else {
            performance_class = "Poor";
        }
        std::cout << "Performance rating: " << performance_class << std::endl;
    }
    
    if (cache) {
        std::cout << "\n--- CACHE PERFORMANCE ANALYSIS ---" << std::endl;
        std::cout << "L1 cache hits: " << cache->l1_hits << std::endl;
        std::cout << "L1 cache misses: " << cache->l1_misses << std::endl;
        std::cout << "L2 cache hits: " << cache->l2_hits << std::endl;
        std::cout << "L2 cache misses: " << cache->l2_misses << std::endl;
        std::cout << "DRAM accesses: " << cache->dram_accesses << std::endl;
        
        std::cout << "\nL1 hit rate: " << std::fixed << std::setprecision(1) 
                  << (cache->l1_hit_rate * 100) << "%" << std::endl;
        std::cout << "L2 hit rate: " << (cache->l2_hit_rate * 100) << "%" << std::endl;
        std::cout << "Average memory latency: " << std::setprecision(1) 
                  << cache->average_memory_latency << " cycles" << std::endl;
        std::cout << "Cache efficiency score: " << std::setprecision(2) 
                  << cache->cache_efficiency << std::endl;
        
        // Cache optimization recommendations
        std::cout << "\n--- OPTIMIZATION RECOMMENDATIONS ---" << std::endl;
        if (cache->l1_hit_rate < 0.8) {
            std::cout << "• Consider improving data locality for L1 cache" << std::endl;
            std::cout << "• Use shared memory for frequently accessed data" << std::endl;
        }
        if (cache->l2_hit_rate < 0.6) {
            std::cout << "• Consider coalesced memory accesses" << std::endl;
            std::cout << "• Optimize memory access patterns" << std::endl;
        }
        if (cache->average_memory_latency > 50.0) {
            std::cout << "• High memory latency detected - consider prefetching" << std::endl;
        }
    }
    
    std::cout << std::string(60, '=') << std::endl;
}

void generate_optimization_report(const KernelOccupancyResult* occupancy, const PowerConsumptionResult* power) {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "           OPTIMIZATION ANALYSIS REPORT" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    
    if (occupancy) {
        std::cout << "\n--- KERNEL OCCUPANCY ANALYSIS ---" << std::endl;
        std::cout << "Max threads per block: " << occupancy->max_threads_per_block << std::endl;
        std::cout << "Max blocks per SM: " << occupancy->max_blocks_per_sm << std::endl;
        std::cout << "Optimal threads per block: " << occupancy->optimal_threads_per_block << std::endl;
        std::cout << "Optimal blocks per SM: " << occupancy->optimal_blocks_per_sm << std::endl;
        
        std::cout << "\nTheoretical occupancy: " << std::fixed << std::setprecision(1) 
                  << (occupancy->theoretical_occupancy * 100) << "%" << std::endl;
        std::cout << "Achieved occupancy: " << (occupancy->achieved_occupancy * 100) << "%" << std::endl;
        std::cout << "Warp efficiency: " << (occupancy->warp_efficiency * 100) << "%" << std::endl;
        std::cout << "Registers per thread: " << occupancy->registers_per_thread << std::endl;
        std::cout << "Shared memory per block: " << (occupancy->shared_memory_per_block / 1024) << " KB" << std::endl;
        
        // Occupancy optimization recommendations
        std::cout << "\n--- OCCUPANCY OPTIMIZATION RECOMMENDATIONS ---" << std::endl;
        if (occupancy->achieved_occupancy < 0.5) {
            std::cout << "• Low occupancy detected - consider reducing register usage" << std::endl;
            std::cout << "• Try different thread block sizes" << std::endl;
        }
        if (occupancy->registers_per_thread > 32) {
            std::cout << "• High register usage may limit occupancy" << std::endl;
            std::cout << "• Consider algorithm modifications to reduce register pressure" << std::endl;
        }
        if (occupancy->shared_memory_per_block > 32*1024) {
            std::cout << "• High shared memory usage may limit occupancy" << std::endl;
        }
        if (occupancy->warp_efficiency < 0.8) {
            std::cout << "• Low warp efficiency - check for thread divergence" << std::endl;
            std::cout << "• Consider restructuring conditional code" << std::endl;
        }
    }
    
    if (power) {
        std::cout << "\n--- POWER CONSUMPTION ANALYSIS ---" << std::endl;
        std::cout << "Idle power: " << std::fixed << std::setprecision(1) << power->idle_power_watts << " W" << std::endl;
        std::cout << "Peak power: " << power->peak_power_watts << " W" << std::endl;
        std::cout << "Average power: " << power->average_power_watts << " W" << std::endl;
        std::cout << "Energy consumed: " << std::setprecision(2) << power->energy_consumed_joules << " J" << std::endl;
        std::cout << "Energy efficiency: " << std::setprecision(3) << power->energy_efficiency_gflops_w << " GFLOPS/W" << std::endl;
        std::cout << "Thermal throttling events: " << (int)power->thermal_throttling_events << std::endl;
        
        // Power efficiency classification
        std::string efficiency_class;
        if (power->energy_efficiency_gflops_w > 10.0) {
            efficiency_class = "Excellent";
        } else if (power->energy_efficiency_gflops_w > 5.0) {
            efficiency_class = "Good";
        } else if (power->energy_efficiency_gflops_w > 2.0) {
            efficiency_class = "Moderate";
        } else {
            efficiency_class = "Poor";
        }
        std::cout << "Energy efficiency rating: " << efficiency_class << std::endl;
        
        // Power optimization recommendations
        std::cout << "\n--- POWER OPTIMIZATION RECOMMENDATIONS ---" << std::endl;
        if (power->average_power_watts / power->peak_power_watts < 0.6) {
            std::cout << "• Low power utilization - consider increasing workload" << std::endl;
        }
        if (power->thermal_throttling_events > 0) {
            std::cout << "• Thermal throttling detected - improve cooling or reduce workload" << std::endl;
        }
        if (power->energy_efficiency_gflops_w < 5.0) {
            std::cout << "• Low energy efficiency - optimize algorithms for better performance/watt" << std::endl;
        }
        
        // Display power timeline if available
        if (!power->power_timeline.empty() && power->power_timeline.size() > 2) {
            std::cout << "\nPower timeline (sample points):" << std::endl;
            for (size_t i = 0; i < std::min(power->power_timeline.size(), (size_t)5); i++) {
                std::cout << "  " << std::setprecision(1) << power->power_timeline[i] << " W";
                if (i < power->temperature_timeline.size()) {
                    std::cout << " (temp: " << (int)power->temperature_timeline[i] << "°C)";
                }
                std::cout << std::endl;
            }
        }
    }
    
    std::cout << std::string(60, '=') << std::endl;
}

void generate_comprehensive_report(EmpiricalAnalysisContext* ctx, const char* output_file) {
    std::ofstream report(output_file);
    if (!report.is_open()) {
        std::cerr << "Error: Cannot create report file " << output_file << std::endl;
        return;
    }
    
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    
    report << "# OBDD CUDA Performance Analysis Report\n";
    report << "Generated on: " << std::ctime(&time_t);
    report << "\n## Device Information\n";
    report << "- Device: " << ctx->device_properties.name << "\n";
    report << "- Compute Capability: " << ctx->device_properties.major << "." << ctx->device_properties.minor << "\n";
    report << "- Global Memory: " << (ctx->device_properties.totalGlobalMem / (1024*1024*1024)) << " GB\n";
    report << "- Multiprocessors: " << ctx->device_properties.multiProcessorCount << "\n";
    report << "- Max Threads per Block: " << ctx->device_properties.maxThreadsPerBlock << "\n";
    report << "- Memory Bus Width: " << ctx->device_properties.memoryBusWidth << " bits\n";
    
    report << "\n## Test Configuration\n";
    report << "- Number of iterations: " << ctx->num_iterations << "\n";
    report << "- Profiling enabled: " << (ctx->enable_profiling ? "Yes" : "No") << "\n";
    report << "- Bandwidth buffer size: " << (ctx->bandwidth_buffer_size / (1024*1024)) << " MB\n";
    
    // Thread counts tested
    if (!ctx->scaling_thread_counts.empty()) {
        report << "- Thread counts tested: ";
        for (size_t i = 0; i < ctx->scaling_thread_counts.size(); i++) {
            if (i > 0) report << ", ";
            report << ctx->scaling_thread_counts[i];
        }
        report << "\n";
    }
    
    // Problem sizes tested  
    if (!ctx->problem_sizes.empty()) {
        report << "- Problem sizes tested: ";
        for (size_t i = 0; i < ctx->problem_sizes.size(); i++) {
            if (i > 0) report << ", ";
            report << ctx->problem_sizes[i];
        }
        report << "\n";
    }
    
    report << "\n## Analysis Summary\n";
    report << "This report contains detailed performance analysis of CUDA OBDD operations\n";
    report << "including scaling behavior, memory utilization, cache performance,\n";
    report << "kernel occupancy optimization, and power consumption characteristics.\n";
    
    report << "\n## Optimization Recommendations\n";
    report << "Based on the empirical analysis, the following optimizations are recommended:\n";
    report << "\n### General Recommendations\n";
    report << "1. Use optimal thread block sizes identified in occupancy analysis\n";
    report << "2. Implement memory access patterns that maximize cache hit rates\n";
    report << "3. Monitor power consumption to prevent thermal throttling\n";
    report << "4. Consider weak scaling limitations for large problem sizes\n";
    report << "5. Profile specific kernels to identify bottlenecks\n";
    
    report << "\n### Memory Optimization\n";
    report << "- Ensure coalesced memory accesses for maximum bandwidth\n";
    report << "- Use shared memory for data reused within thread blocks\n";
    report << "- Consider prefetching for irregular access patterns\n";
    
    report << "\n### Compute Optimization\n";  
    report << "- Balance occupancy vs. cache utilization\n";
    report << "- Minimize thread divergence in conditional code\n";
    report << "- Use appropriate floating-point precision\n";
    
    report << "\n---\n";
    report << "Report generated by OBDD CUDA Empirical Analysis Framework\n";
    
    report.close();
    
    std::cout << "\nComprehensive report saved to: " << output_file << std::endl;
}

/* =====================================================
   PERFORMANCE COMPARISON FUNCTIONS
   ===================================================== */

void compare_optimization_impact(
    const MemoryBandwidthResult* baseline_bandwidth,
    const MemoryBandwidthResult* optimized_bandwidth,
    const CacheMissResult* baseline_cache,
    const CacheMissResult* optimized_cache) {
    
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "          OPTIMIZATION IMPACT COMPARISON" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    
    if (baseline_bandwidth && optimized_bandwidth) {
        double bandwidth_improvement = 
            (optimized_bandwidth->achieved_bandwidth_gbps - baseline_bandwidth->achieved_bandwidth_gbps) /
            baseline_bandwidth->achieved_bandwidth_gbps;
        
        std::cout << "\n--- BANDWIDTH IMPROVEMENTS ---" << std::endl;
        std::cout << "Baseline: " << std::fixed << std::setprecision(1) 
                  << baseline_bandwidth->achieved_bandwidth_gbps << " GB/s" << std::endl;
        std::cout << "Optimized: " << optimized_bandwidth->achieved_bandwidth_gbps << " GB/s" << std::endl;
        std::cout << "Improvement: " << std::showpos << std::setprecision(1) 
                  << (bandwidth_improvement * 100) << "%" << std::noshowpos << std::endl;
    }
    
    if (baseline_cache && optimized_cache) {
        double l1_improvement = optimized_cache->l1_hit_rate - baseline_cache->l1_hit_rate;
        double l2_improvement = optimized_cache->l2_hit_rate - baseline_cache->l2_hit_rate;
        
        std::cout << "\n--- CACHE IMPROVEMENTS ---" << std::endl;
        std::cout << "L1 hit rate improvement: " << std::showpos << std::setprecision(1) 
                  << (l1_improvement * 100) << "%" << std::noshowpos << std::endl;
        std::cout << "L2 hit rate improvement: " << std::showpos << std::setprecision(1) 
                  << (l2_improvement * 100) << "%" << std::noshowpos << std::endl;
        
        double latency_reduction = 
            (baseline_cache->average_memory_latency - optimized_cache->average_memory_latency) /
            baseline_cache->average_memory_latency;
        std::cout << "Memory latency reduction: " << std::showpos << std::setprecision(1) 
                  << (latency_reduction * 100) << "%" << std::noshowpos << std::endl;
    }
    
    std::cout << std::string(60, '=') << std::endl;
}

void generate_performance_summary_table(
    const StrongScalingResult* strong,
    const MemoryBandwidthResult* bandwidth,
    const KernelOccupancyResult* occupancy,
    const PowerConsumptionResult* power) {
    
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "                    PERFORMANCE SUMMARY TABLE" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    
    std::cout << std::left << std::setw(30) << "Metric" 
              << std::setw(15) << "Value" 
              << std::setw(10) << "Unit"
              << std::setw(25) << "Performance Rating" << std::endl;
    std::cout << std::string(80, '-') << std::endl;
    
    if (strong && !strong->speedups.empty()) {
        auto max_speedup = *std::max_element(strong->speedups.begin(), strong->speedups.end());
        std::string rating = (max_speedup > 8) ? "Excellent" : 
                           (max_speedup > 4) ? "Good" : 
                           (max_speedup > 2) ? "Moderate" : "Poor";
        std::cout << std::setw(30) << "Max Strong Scaling Speedup"
                  << std::setw(15) << std::fixed << std::setprecision(1) << max_speedup
                  << std::setw(10) << "x"
                  << std::setw(25) << rating << std::endl;
    }
    
    if (bandwidth) {
        std::string rating = (bandwidth->bandwidth_utilization > 0.8) ? "Excellent" : 
                           (bandwidth->bandwidth_utilization > 0.6) ? "Good" : 
                           (bandwidth->bandwidth_utilization > 0.4) ? "Moderate" : "Poor";
        std::cout << std::setw(30) << "Memory Bandwidth Utilization"
                  << std::setw(15) << std::fixed << std::setprecision(1) << (bandwidth->bandwidth_utilization * 100)
                  << std::setw(10) << "%"
                  << std::setw(25) << rating << std::endl;
        
        std::cout << std::setw(30) << "Achieved Bandwidth"
                  << std::setw(15) << std::setprecision(1) << bandwidth->achieved_bandwidth_gbps
                  << std::setw(10) << "GB/s"
                  << std::setw(25) << "N/A" << std::endl;
    }
    
    if (occupancy) {
        std::string rating = (occupancy->achieved_occupancy > 0.75) ? "Excellent" : 
                           (occupancy->achieved_occupancy > 0.5) ? "Good" : 
                           (occupancy->achieved_occupancy > 0.25) ? "Moderate" : "Poor";
        std::cout << std::setw(30) << "Kernel Occupancy"
                  << std::setw(15) << std::fixed << std::setprecision(1) << (occupancy->achieved_occupancy * 100)
                  << std::setw(10) << "%"
                  << std::setw(25) << rating << std::endl;
    }
    
    if (power) {
        std::string rating = (power->energy_efficiency_gflops_w > 10) ? "Excellent" : 
                           (power->energy_efficiency_gflops_w > 5) ? "Good" : 
                           (power->energy_efficiency_gflops_w > 2) ? "Moderate" : "Poor";
        std::cout << std::setw(30) << "Energy Efficiency"
                  << std::setw(15) << std::fixed << std::setprecision(2) << power->energy_efficiency_gflops_w
                  << std::setw(10) << "GFLOPS/W"
                  << std::setw(25) << rating << std::endl;
        
        std::cout << std::setw(30) << "Average Power Consumption"
                  << std::setw(15) << std::setprecision(1) << power->average_power_watts
                  << std::setw(10) << "W"
                  << std::setw(25) << "N/A" << std::endl;
    }
    
    std::cout << std::string(80, '=') << std::endl;
}

#endif /* OBDD_ENABLE_CUDA */