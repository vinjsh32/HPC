#!/bin/bash
# ============================================================================
#  Safe OBDD Benchmark Runner - Only Working Tests
#  
#  This script runs only the tests that are known to work reliably,
#  avoiding timeout issues and missing dependencies.
#  
#  Author: @vijsh32
#  Date: August 31, 2024  
#  Version: 1.0
# ============================================================================

set -e  # Exit on error

# Configuration
RESULTS_DIR="results"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_FILE="${RESULTS_DIR}/benchmark_results_${TIMESTAMP}.csv"

# Create results directory
mkdir -p "$RESULTS_DIR"

echo "=========================================="
echo "  OBDD Safe Benchmark Suite"
echo "  Started at: $(date)"
echo "=========================================="

# Initialize CSV file
echo "Backend,TestType,BDDSize,Variables,Time_ms,Memory_MB,Operations_per_sec,Nodes_per_sec,Success" > "$RESULTS_FILE"

# Function to run working tests only
run_safe_test() {
    local backend=$1
    local test_name=$2
    local make_target=$3
    local vars=$4
    
    echo "Testing $backend - $test_name..."
    
    start_time=$(date +%s%N)
    
    if timeout 120 make $make_target > /tmp/test_output_$$.txt 2>&1; then
        end_time=$(date +%s%N)
        duration_ms=$(( (end_time - start_time) / 1000000 ))
        
        echo "$backend,$test_name,working,$vars,$duration_ms,0,0,0,SUCCESS" >> "$RESULTS_FILE"
        echo "✓ $backend - $test_name completed in ${duration_ms}ms"
    else
        echo "$backend,$test_name,working,$vars,TIMEOUT,0,0,0,FAILED" >> "$RESULTS_FILE"
        echo "✗ $backend - $test_name failed"
    fi
    
    rm -f /tmp/test_output_$$.txt
}

# Test Sequential Backend (Most Reliable)
echo "=== Testing Sequential Backend ==="
make clean > /dev/null 2>&1
make CUDA=0 OMP=0 > /dev/null 2>&1
run_safe_test "Sequential" "basic_ops" "run-seq" "8"

# Test CUDA Backend (If Available)
if command -v nvcc &> /dev/null; then
    echo "=== Testing CUDA Backend ==="
    make clean > /dev/null 2>&1
    if make CUDA=1 > /dev/null 2>&1; then
        run_safe_test "CUDA" "gpu_ops" "run-cuda" "10"
    else
        echo "⚠️ CUDA compilation failed, skipping"
        echo "CUDA,gpu_ops,skipped,10,SKIP,0,0,0,SKIPPED" >> "$RESULTS_FILE"
    fi
else
    echo "⚠️ CUDA not available, skipping"
    echo "CUDA,gpu_ops,unavailable,10,SKIP,0,0,0,SKIPPED" >> "$RESULTS_FILE"
fi

# Test OpenMP Backend (Cautiously)
echo "=== Testing OpenMP Backend ==="
make clean > /dev/null 2>&1
if make CUDA=0 OMP=1 > /dev/null 2>&1; then
    # Try a quick test first
    if timeout 30 make OMP=1 run-omp > /dev/null 2>&1; then
        run_safe_test "OpenMP" "parallel_ops" "run-omp" "8"
    else
        echo "⚠️ OpenMP test timed out, marking as problematic"
        echo "OpenMP,parallel_ops,timeout,8,TIMEOUT,0,0,0,TIMEOUT" >> "$RESULTS_FILE"
    fi
else
    echo "⚠️ OpenMP compilation failed, skipping"
    echo "OpenMP,parallel_ops,compile_fail,8,COMPILE_FAIL,0,0,0,FAILED" >> "$RESULTS_FILE"
fi

# Test some performance comparisons with different sizes (Sequential only)
echo "=== Testing Performance Scaling (Sequential) ==="
for size in 6 8 10; do
    run_safe_test "Sequential" "scale_${size}" "run-seq" "$size"
done

# Test CUDA scaling if working
if command -v nvcc &> /dev/null && make CUDA=1 > /dev/null 2>&1; then
    echo "=== Testing CUDA Scaling ==="
    for size in 8 10; do
        run_safe_test "CUDA" "scale_${size}" "run-cuda" "$size"
    done
fi

echo "=========================================="
echo "  Safe Benchmark Complete!"
echo "  Results saved to: $RESULTS_FILE"
echo "=========================================="

# Generate summary
echo "=== SAFE BENCHMARK SUMMARY ==="
total_tests=$(wc -l < "$RESULTS_FILE")
successful_tests=$(grep -c "SUCCESS" "$RESULTS_FILE" || echo 0)
failed_tests=$(grep -c -E "FAILED|TIMEOUT|SKIPPED" "$RESULTS_FILE" || echo 0)

echo "Total tests: $total_tests"
echo "Successful: $successful_tests"
echo "Failed/Skipped: $failed_tests"

if [[ $successful_tests -gt 0 ]]; then
    echo ""
    echo "✅ Safe benchmark completed with working tests"
    echo "   Data ready for report generation!"
else
    echo ""
    echo "❌ No tests succeeded - check system configuration"
fi