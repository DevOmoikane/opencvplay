#!/bin/bash

# Script to find and delete duplicate files from first directory that exist in second directory

show_usage() {
    echo "Usage: $0 <source_directory> <reference_directory>"
    echo "  This script will delete files from source_directory that also exist in reference_directory"
    echo "  Warning: This permanently deletes files!"
    exit 1
}

# Check if both directories are provided
if [ $# -ne 2 ]; then
    echo "Error: Please provide two directory paths"
    show_usage
fi

SOURCE_DIR="$1"
REF_DIR="$2"

# Check if directories exist
if [ ! -d "$SOURCE_DIR" ]; then
    echo "Error: Source directory '$SOURCE_DIR' does not exist"
    exit 1
fi

if [ ! -d "$REF_DIR" ]; then
    echo "Error: Reference directory '$REF_DIR' does not exist"
    exit 1
fi

# Check if directories are the same
if [ "$SOURCE_DIR" = "$REF_DIR" ]; then
    echo "Error: Source and reference directories are the same!"
    exit 1
fi

echo "Comparing directories:"
echo "  Source: $SOURCE_DIR"
echo "  Reference: $REF_DIR"
echo

# Find duplicate files
echo "Looking for duplicate files..."
DUPLICATES=()
DELETED_COUNT=0
SAFE_COUNT=0

# Loop through all files in source directory
while IFS= read -r -d '' file; do
    filename=$(basename "$file")
    ref_file="$REF_DIR/$filename"

    # Check if file exists in reference directory
    if [ -f "$ref_file" ]; then
        # Optional: Compare file contents to be sure they're identical
        # Uncomment the next lines if you want content comparison
        # if cmp -s "$file" "$ref_file"; then
        #     DUPLICATES+=("$file")
        # fi

        DUPLICATES+=("$file")
    fi
done < <(find "$SOURCE_DIR" -maxdepth 1 -type f -print0)

if [ ${#DUPLICATES[@]} -eq 0 ]; then
    echo "No duplicate files found."
    exit 0
fi

echo "Found ${#DUPLICATES[@]} duplicate file(s):"
printf '  %s\n' "${DUPLICATES[@]}"
echo

# Ask for confirmation before deletion
read -p "Do you want to delete these ${#DUPLICATES[@]} file(s) from '$SOURCE_DIR'? (y/N): " -n 1 -r
echo

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Operation cancelled."
    exit 0
fi

echo
echo "Deleting duplicate files..."

# Delete the duplicate files
for file in "${DUPLICATES[@]}"; do
    if rm "$file"; then
        echo "Deleted: $file"
        ((DELETED_COUNT++))
    else
        echo "Error deleting: $file"
        ((SAFE_COUNT++))
    fi
done

echo
echo "Operation completed:"
echo "  Successfully deleted: $DELETED_COUNT file(s)"
echo "  Failed to delete: $SAFE_COUNT file(s)"

# Show remaining files in source directory
echo
echo "Remaining files in source directory:"
find "$SOURCE_DIR" -maxdepth 1 -type f -printf "  %f\n" | sort
