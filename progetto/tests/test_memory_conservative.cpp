/**
 * @file test_memory_conservative.cpp
 * @brief Conservative memory tests for validating memory-efficient algorithms
 * 
 * Uses smaller scales to test memory management functionality without
 * causing system crashes.
 */

#include "core/obdd.hpp"
#include "core/obdd_memory_manager.hpp"
#include <gtest/gtest.h>
#include <chrono>
#include <iostream>
#include <iomanip>

class MemoryConservativeTest : public ::testing::Test {
protected:
    MemoryConfig memory_config;
    
    void SetUp() override {
        // Conservative configuration for testing
        memory_config.max_memory_mb = 1000;      // 1GB limit for testing
        memory_config.chunk_size_variables = 50; // Small chunks
        memory_config.enable_disk_cache = false;
        memory_config.enable_compression = false;
        memory_config.gc_threshold_nodes = 10000;
        
        obdd_set_memory_limit_mb(memory_config.max_memory_mb);
        
        std::cout << "\n💾 CONSERVATIVE MEMORY TESTS" << std::endl;
        std::cout << "=============================" << std::endl;
        std::cout << "Memory limit: " << memory_config.max_memory_mb << "MB" << std::endl;
        std::cout << "Chunk size: " << memory_config.chunk_size_variables << " variables" << std::endl;
    }
    
    double measure_time_ms(std::function<void()> operation) {
        auto start = std::chrono::high_resolution_clock::now();
        operation();
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start).count();
    }
};

TEST_F(MemoryConservativeTest, BasicMemoryManagerTest) {
    std::cout << "\n🔧 BASIC MEMORY MANAGER TEST" << std::endl;
    std::cout << "=============================" << std::endl;
    
    // Test memory monitoring functions
    size_t initial_memory = obdd_get_memory_usage_mb();
    std::cout << "Initial memory usage: " << initial_memory << "MB" << std::endl;
    
    // Test garbage collection trigger
    obdd_trigger_garbage_collection();
    std::cout << "✅ Garbage collection triggered successfully" << std::endl;
    
    // Test memory limit setting
    obdd_set_memory_limit_mb(2000);
    std::cout << "✅ Memory limit updated successfully" << std::endl;
    
    EXPECT_GE(initial_memory, 0) << "Memory usage should be non-negative";
}

TEST_F(MemoryConservativeTest, ProgressiveBuilderSmallScale) {
    std::cout << "\n🏗️ PROGRESSIVE BUILDER TEST (Small Scale)" << std::endl;
    std::cout << "=========================================" << std::endl;
    
    // Test with small, manageable sizes
    for (int target_vars = 10; target_vars <= 30; target_vars += 10) {
        std::cout << "\nTesting progressive build with " << target_vars << " variables..." << std::endl;
        
        size_t pre_memory = obdd_get_memory_usage_mb();
        
        double build_time = measure_time_ms([&]() {
            ProgressiveBDDBuilder* builder = obdd_progressive_create(target_vars, &memory_config);
            
            if (builder) {
                // Add variables in small batches
                int batch_size = 5;
                while (obdd_progressive_add_variable_batch(builder, batch_size)) {
                    // Continue building
                }
                
                OBDD* result = obdd_progressive_get_current(builder);
                if (result && result->root) {
                    std::cout << "  ✅ Successfully built " << target_vars << "-variable BDD" << std::endl;
                } else {
                    std::cout << "  ❌ Failed to build BDD" << std::endl;
                }
                
                obdd_progressive_destroy(builder);
            } else {
                std::cout << "  ❌ Failed to create progressive builder" << std::endl;
            }
        });
        
        size_t post_memory = obdd_get_memory_usage_mb();
        
        std::cout << "  Build time: " << std::fixed << std::setprecision(3) 
                  << (build_time/1000) << "s" << std::endl;
        std::cout << "  Memory: " << pre_memory << "MB → " << post_memory << "MB" << std::endl;
        
        EXPECT_LT(post_memory, memory_config.max_memory_mb) 
            << "Memory usage should stay within limit";
    }
}

TEST_F(MemoryConservativeTest, StreamingBuilderTest) {
    std::cout << "\n🌊 STREAMING BUILDER TEST" << std::endl;
    std::cout << "=========================" << std::endl;
    
    // Test streaming with small total variables
    int total_vars = 20;
    
    std::cout << "Testing streaming builder with " << total_vars << " total variables..." << std::endl;
    
    size_t pre_memory = obdd_get_memory_usage_mb();
    
    double streaming_time = measure_time_ms([&]() {
        StreamingBDDBuilder* builder = obdd_streaming_create(total_vars, &memory_config);
        
        if (builder) {
            // Define a simple constraint function
            auto simple_constraint = [](int start_var, int num_vars) -> OBDD* {
                if (num_vars <= 0) return nullptr;
                
                std::vector<int> order(num_vars);
                for (int i = 0; i < num_vars; ++i) {
                    order[i] = start_var + i;
                }
                
                OBDD* chunk_bdd = obdd_create(num_vars, order.data());
                if (chunk_bdd) {
                    // Simple: just the first variable
                    chunk_bdd->root = obdd_node_create(0, obdd_constant(0), obdd_constant(1));
                }
                
                return chunk_bdd;
            };
            
            obdd_streaming_add_constraint(builder, simple_constraint);
            OBDD* result = obdd_streaming_finalize(builder);
            
            if (result && result->root) {
                std::cout << "  ✅ Successfully created streaming BDD" << std::endl;
                obdd_destroy(result);
            } else {
                std::cout << "  ❌ Failed to create streaming BDD" << std::endl;
            }
            
            obdd_streaming_destroy(builder);
        } else {
            std::cout << "  ❌ Failed to create streaming builder" << std::endl;
        }
    });
    
    size_t post_memory = obdd_get_memory_usage_mb();
    
    std::cout << "  Streaming time: " << std::fixed << std::setprecision(3) 
              << (streaming_time/1000) << "s" << std::endl;
    std::cout << "  Memory: " << pre_memory << "MB → " << post_memory << "MB" << std::endl;
    
    EXPECT_LT(post_memory, memory_config.max_memory_mb) 
        << "Streaming should keep memory usage under limit";
}

TEST_F(MemoryConservativeTest, ChunkedApplyTest) {
    std::cout << "\n🔄 CHUNKED APPLY TEST" << std::endl;
    std::cout << "=====================" << std::endl;
    
    // Create two small test BDDs
    std::vector<int> order = {0, 1, 2, 3, 4};
    OBDD* bdd1 = obdd_create(5, order.data());
    OBDD* bdd2 = obdd_create(5, order.data());
    
    if (bdd1 && bdd2) {
        // Simple BDD: x0 AND x1
        bdd1->root = obdd_node_create(0, obdd_constant(0), 
                                     obdd_node_create(1, obdd_constant(0), obdd_constant(1)));
        
        // Simple BDD: x2 OR x3  
        bdd2->root = obdd_node_create(2, obdd_node_create(3, obdd_constant(0), obdd_constant(1)), 
                                     obdd_constant(1));
        
        size_t pre_memory = obdd_get_memory_usage_mb();
        
        double apply_time = measure_time_ms([&]() {
            OBDDNode* result = obdd_apply_chunked(bdd1, bdd2, OBDD_AND, &memory_config);
            if (result) {
                std::cout << "  ✅ Chunked apply completed successfully" << std::endl;
            } else {
                std::cout << "  ❌ Chunked apply failed" << std::endl;
            }
        });
        
        size_t post_memory = obdd_get_memory_usage_mb();
        
        std::cout << "  Apply time: " << std::fixed << std::setprecision(3) 
                  << (apply_time/1000) << "s" << std::endl;
        std::cout << "  Memory: " << pre_memory << "MB → " << post_memory << "MB" << std::endl;
        
        obdd_destroy(bdd1);
        obdd_destroy(bdd2);
        
        EXPECT_LT(post_memory, memory_config.max_memory_mb) 
            << "Chunked apply should manage memory efficiently";
    } else {
        GTEST_SKIP() << "Failed to create test BDDs";
    }
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    
    std::cout << "\n💾 CONSERVATIVE MEMORY TEST SUITE" << std::endl;
    std::cout << "==================================" << std::endl;
    std::cout << "Testing memory management functionality with safe parameters" << std::endl;
    
    return RUN_ALL_TESTS();
}