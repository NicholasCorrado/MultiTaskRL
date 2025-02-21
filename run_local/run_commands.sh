#!/bin/bash

# Check if a filename argument is provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <commands_file>"
    exit 1
fi

# Assign the first argument to INPUT_FILE
INPUT_FILE="chtc/commands/$1"

# Check if the file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: File '$INPUT_FILE' not found!"
    exit 1
fi

# Read and execute each line
while IFS= read -r command; do
    # Ignore empty lines and comments
    [[ -z "$command" || "$command" =~ ^#.*$ ]] && continue

    echo "Executing: $command"
    eval "$command"
done < "$INPUT_FILE"

