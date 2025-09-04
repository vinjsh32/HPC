#!/bin/bash
# Script to add GPLv3 license headers to source files

echo "Adding GPLv3 license headers to source files..."

# Function to add header to a file
add_header_to_file() {
    local file="$1"
    local temp_file="${file}.tmp"
    
    # Check if file already has GPLv3 license
    if grep -q "GNU General Public License" "$file"; then
        echo "License header already exists in: $file"
        return 0
    fi
    
    echo "Adding license header to: $file"
    
    # Create temporary file with header + original content
    cat > "$temp_file" << 'EOF'
/*
 * This file is part of the High-Performance OBDD Library
 * Copyright (C) 2024 High Performance Computing Laboratory
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <https://www.gnu.org/licenses/>.
 * 
 * Authors: vinjsh32
 * Student ID: [Student ID]
 * Email: [Student Email]
 * Assignment: Final Project - Parallel OBDD Implementation
 * Course: High Performance Computing - Prof. Moscato
 * University: Università degli studi di Salerno - Ingegneria Informatica magistrale
 */

EOF
    
    # Append original file content
    cat "$file" >> "$temp_file"
    
    # Replace original with new version
    mv "$temp_file" "$file"
    
    echo "License header added to: $file"
}

# Process all source files
for file in $(find ../include ../src -name "*.cpp" -o -name "*.hpp" -o -name "*.cu" -o -name "*.h" 2>/dev/null); do
    add_header_to_file "$file"
done

echo "License header addition completed!"