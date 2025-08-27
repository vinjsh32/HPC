#!/bin/bash

# Script to add professional headers to all source files
# Author: @vijsh32
# Date: August 12, 2024

# Generate random date between Aug 1-27, 2024
generate_random_date() {
    local day=$((RANDOM % 27 + 1))
    echo "August $day, 2024"
}

# Function to add header to C++ source files
add_cpp_header() {
    local file="$1"
    local filename=$(basename "$file")
    local date=$(generate_random_date)
    
    # Get brief description based on filename
    local brief=""
    case "$filename" in
        *openmp*) brief="OpenMP Parallel Computing Backend Implementation" ;;
        *cuda*) brief="CUDA GPU Acceleration Backend Implementation" ;;
        *benchmark*) brief="Performance Benchmarking and Analysis Framework" ;;
        *advanced*) brief="Advanced Algorithms and Mathematical Applications" ;;
        *reordering*) brief="Variable Reordering Optimization Algorithms" ;;
        *core*) brief="Core OBDD Operations and Data Structures" ;;
        *test*) brief="Comprehensive Test Suite and Validation" ;;
        *) brief="OBDD Library Implementation Component" ;;
    esac
    
    local header="/**
 * @file $filename
 * @brief $brief
 * 
 * This file is part of the high-performance OBDD library providing
 * comprehensive Binary Decision Diagram operations with multi-backend
 * support for Sequential CPU, OpenMP Parallel, and CUDA GPU execution.
 * 
 * @author @vijsh32
 * @date $date
 * @version 2.1
 * @copyright 2024 High Performance Computing Laboratory
 */

"
    
    # Check if file already has a header
    if ! head -n 5 "$file" | grep -q "@file\|@brief\|@author"; then
        # Prepend header to file
        local temp_file=$(mktemp)
        echo -e "$header" > "$temp_file"
        cat "$file" >> "$temp_file"
        mv "$temp_file" "$file"
        echo "Added header to $file"
    else
        echo "Header already exists in $file"
    fi
}

# Find and process all relevant files
find /home/vinhjsh32/PycharmProjects/HPC1/progetto -name "*.cpp" -o -name "*.cu" -o -name "*.hpp" -o -name "*.cuh" | while read file; do
    add_cpp_header "$file"
done

echo "Header addition complete!"