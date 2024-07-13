#!/bin/bash

# Define arrays of hyperparameter values
sparsity_fracs=(0.90 0.99)
leaks=(0.1 0.001)
block_overlap_depths=(1 2)
act_regularizations=(0.1 0.01 0.001 0)
weight_regularizations=(0.1 0.01 0.001 0)
num_steps_per_frames=(1 5)

# Get current date with seconds and AOE timezone -- used to disambiguate runs
current_date=$(date +"%Y-%m-%d_%H-%M-%S_%Z" -u)

# Base output directory for results
output_dir="results/hyperparam_search_${current_date}"

# constants
data_dir="dataset/k400/train"
val_dir="dataset/debug_val"
batch_size=10
visualization_period=100
val_period=100
lr=0.001
min_length=64
max_length=128
min_num_blocks=2
max_num_blocks=5
video_width=50
video_height=50
num_epochs=500


# Print the result
echo "$current_date"



# Create output directory if it doesn't exist
mkdir -p $output_dir

# Generate commands for each combination of hyperparameters
for sparsity_frac in "${sparsity_fracs[@]}"
do
    for leak in "${leaks[@]}"
    do
        for block_overlap_depth in "${block_overlap_depths[@]}"
        do
            for act_regularization in "${act_regularizations[@]}"
            do
                for weight_regularization in "${weight_regularizations[@]}"
                do
                    for num_steps_per_frame in "${num_steps_per_frames[@]}"
                    do
                        echo "python3 cortex_cube/main.py --sparsity_frac $sparsity_frac --leak $leak --block_overlap_depth $block_overlap_depth --activity_regularization $act_regularization --weight_regularization $weight_regularization --num_steps_per_frame $num_steps_per_frame --out_dir $output_dir/sparsity_${sparsity_frac}_leak_${leak}_block_${block_overlap_depth}_actreg_${act_regularization}_weightreg_${weight_regularization}_steps_${num_steps_per_frame} --batch_size $batch_size --visualization_period $visualization_period --val_period $val_period --lr $lr --min_length $min_length --max_length $max_length --min_num_blocks $min_num_blocks --max_num_blocks $max_num_blocks --video_width $video_width --video_height $video_height --data_dir $data_dir --val_dir $val_dir --num_epochs $num_epochs" >> scripts/${current_date}run_commands.txt
                    done
                done
            done
        done
    done
done

# Print instructions
echo "Commands to run the experiments have been generated in scripts/${current_date}run_commands.txt."
echo "You can run these commands manually on different machines or use a job scheduler like GNU Parallel or a cluster job submission system to run them in parallel."
