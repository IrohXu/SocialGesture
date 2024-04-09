#!/bin/bash

# Define your directory path here. Replace "/path/to/your/folder" with the actual path.
DIRECTORY="/home/xucao2/tool/SocialGesture/data_sources"

# Find files in the directory (including subdirectories) and rename them
# Loop through all files in the directory
find "$DIRECTORY" -type f -name "* *" | while read file; do
    # Replace spaces with underscores in the file name
    newname=$(echo "$file" | tr ' ' '_')
    # Move the file to the new name
    mv "$file" "$newname"
done