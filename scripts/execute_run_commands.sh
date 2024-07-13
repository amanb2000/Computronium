#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 start_index end_index"
    exit 1
fi

start_idx=$1
end_idx=$2
command_file="run_commands.txt"

# Check if the command file exists
if [ ! -f "$command_file" ]; then
    echo "Error: $command_file does not exist."
    exit 1
fi

# Extract and run commands within the specified range
awk "NR >= $start_idx && NR <= $end_idx" $command_file | while read -r command
do
    echo "Executing: $command"
    eval $command
done

echo "Completed running commands from line $start_idx to $end_idx in $command_file."
