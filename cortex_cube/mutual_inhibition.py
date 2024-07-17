"""
Usage (get commandline args help): 
python3 cortex_cube/mutual_inhibition.py --help 

Here we will train two computronium cubes that attempt to mutually inhibit each 
other. The "left" cube will try to send the shared layer activations to zero, 
while the "right" cube will try to send the shared layer activations to one. 
"""

import os
import json 
import argparse
from datetime import datetime
from cube_model import cube

import torch 
import torch.nn as nn
import torch.optim as optim


def log(msg, file_path): 
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(file_path, 'a') as f: 
        f.write(f"[{current_time}] {msg}\n")

def parse_args():
    parser = argparse.ArgumentParser(description="Train two cubes that mutually inhibit each other's activity in a shared set of layers.")
    # Arguments for cube architecture: 
    # --dt, --num_blocks_left, --num_blocks_right, --internal_block_overlap_depth, --height, --width, --kernel_size, --num_channels_list
    parser.add_argument("--dt", type=float, default=0.01, help="Timestep for simulation. Default=0.01.")
    parser.add_argument("--timesteps_per_comparison", type=int, default=1, help="Number of timesteps to run each block between comparisons of shared layer activity. Default=1.")
    parser.add_argument("--num_blocks_left", type=int, default=3, help="Number of blocks in the left cube. Default=3.")
    parser.add_argument("--num_blocks_right", type=int, default=3, help="Number of blocks in the right cube. Default=3.")
    parser.add_argument("--left_obs_sparsity_frac", type=float, default=0.1, help="Fraction of observed units in the left cube (0-1, 1=max sparsity). Default=0.1.")
    parser.add_argument("--right_obs_sparsity_frac", type=float, default=0.1, help="Fraction of observed units in the right cube (0-1, 1=max sparsity). Default=0.1.")
    parser.add_argument("--internal_block_overlap_depth", type=int, default=1, help="Number of layers that overlap between adjacent blocks. Default=1.")
    parser.add_argument("--height", type=int, default=256, help="Height of the cube. Default=256.")
    parser.add_argument("--width", type=int, default=32, help="Width of the cube. Default=256")
    parser.add_argument("--kernel_size", type=int, default=3, help="Size of the convolutional kernels. Default=3.")
    parser.add_argument("--num_channels_list", type=int, nargs="+", default=[16, 32, 64], help="Number of channels in each convolutional layer. Default=[16, 32, 64].")

    # Arguments for training:
    # --lr_left, --lr_right, --num_epochs, --num_timesteps, --batch_size
    parser.add_argument("--lr_left", type=float, default=0.01, help="Learning rate for the left cube.")
    parser.add_argument("--lr_right", type=float, default=0.01, help="Learning rate for the right cube.")
    parser.add_argument("--num_epochs", type=int, default=1000, help="Number of epochs to train for.")
    parser.add_argument("--num_timesteps", type=int, default=100, help="Number of timesteps to simulate for each training example.")
    parser.add_argument("--leak", type=float, default=0.1, help="Per-timestep leak activity[t+1] = activity[t]*(1-leak) for computronium cubes (0-1). Default=0.1.")


    # Arguments for saving:
    # --save_dir, --save_every
    parser.add_argument("--save_dir", type=str, default="results/MI/test00", help="Directory to save the trained models.")
    parser.add_argument("--save_every", type=int, default=20, help="Save the models and shared layer videos every n epochs. Default=20.")

def setup_save_dir(args): 
    # make args.save_dir if it doesn't exist
    if not os.path.exists(args.save_dir): 
        os.makedirs(args.save_dir)
    # save args.json in args.save_dir
    with open(os.path.join(args.save_dir, "args.json"), "w") as f: 
        json.dump(vars(args), f)


def main(): 
    args = parse_args()
    setup_save_dir(args)

    # get YYYYMMDD_HHMMSS in California
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(args.save_dir, f"log{current_time}.log")
    print(f"Saving logs to {log_path}")

    log("Starting training.", log_path)

    if not args.mps: 
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else: 
        device = torch.device("mps:0")
    print("Device: ", device)
    log("Device: " + str(device), log_path)

    log("Making left model...")
    left_model = cube(
        kernel_size = args.kernel_size,
        num_channels_list = args.num_channels_list, 
        block_overlap_depth = args.internal_block_overlap_depth,
        device = device, 
        leak = args.leak, 
        dt = args.dt, 
        sparsity_frac=args.left_obs_sparsity_frac
    ).to(device)
    log("Done making left model.")

    log("Making right model...")
    right_model = cube(
        kernel_size = args.kernel_size,
        num_channels_list = args.num_channels_list, 
        block_overlap_depth = args.internal_block_overlap_depth,
        device = device, 
        leak = args.leak, 
        dt = args.dt, 
        sparsity_frac=args.right_obs_sparsity_frac
    ).to(device)
    log("Done making right model.")


    # set up optimizer 
    optimizer = optim.Adam(model.parameters(), lr=args.lr)



if __name__ == "__main__": 
    main()