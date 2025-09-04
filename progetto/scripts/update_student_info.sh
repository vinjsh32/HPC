#!/bin/bash
# Script to update student information in all license headers

echo "Updating student information in all source files..."

# Function to update student info in a file
update_student_info() {
    local file="$1"
    
    if grep -q "GNU General Public License" "$file"; then
        echo "Updating student info in: $file"
        
        # Create backup
        cp "$file" "${file}.bak"
        
        # Update the student information
        sed -i 's/Authors: vinjsh32/Authors: Vincenzo Ferraro/' "$file"
        sed -i 's/Student ID: \[Student ID\]/Student ID: 0622702113/' "$file"
        sed -i 's/Email: \[Student Email\]/Email: v.ferraro5@studenti.unisa.it/' "$file"
        
        echo "Updated: $file"
        
        # Remove backup if successful
        rm "${file}.bak"
    fi
}

# Find all source files and update them
find ../include ../src -name "*.cpp" -o -name "*.hpp" -o -name "*.cu" -o -name "*.h" | while read file; do
    update_student_info "$file"
done

echo "Student information update completed!"
echo ""
echo "Updated information:"
echo "- Name/Surname: Vincenzo Ferraro"
echo "- Student ID: 0622702113" 
echo "- Email: v.ferraro5@studenti.unisa.it"