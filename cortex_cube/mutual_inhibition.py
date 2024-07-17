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
    parser.add_argument("--num_channels_list", type=int, nargs="+", default=[16, 64, 16], help="Number of channels in each convolutional layer. Default=[16, 32, 64].")

    # Arguments for training:
    # --lr_left, --lr_right, --num_epochs, --num_timesteps, --batch_size
    parser.add_argument("--lr_left", type=float, default=0.01, help="Learning rate for the left cube.")
    parser.add_argument("--lr_right", type=float, default=0.01, help="Learning rate for the right cube.")
    parser.add_argument("--num_epochs", type=int, default=1000, help="Number of epochs to train for.")
    parser.add_argument("--num_timesteps", type=int, default=100, help="Number of timesteps to simulate for each training example.")
    parser.add_argument("--batch_size", type=int, default=5, help="Batch size for training. Default=5.")
    parser.add_argument("--leak", type=float, default=0.1, help="Per-timestep leak activity[t+1] = activity[t]*(1-leak) for computronium cubes (0-1). Default=0.1.")


    # Arguments for saving:
    # --save_dir, --save_every
    parser.add_argument("--save_dir", type=str, default="results/MI/test00", help="Directory to save the trained models.")
    parser.add_argument("--save_every", type=int, default=20, help="Save the models and shared layer videos every n epochs. Default=20.")

    parser.add_argument("--mps", action="store_true", help="Use MPS (matrix product state) for training. Default=False.")

    args = parser.parse_args()
    return args

def setup_save_dir(args): 
    # make args.save_dir if it doesn't exist
    if not os.path.exists(args.save_dir): 
        os.makedirs(args.save_dir)
    # save args.json in args.save_dir
    with open(os.path.join(args.save_dir, "args.json"), "w") as f: 
        json.dump(vars(args), f)


def get_inhibition_loss(left_model:cube, 
                        right_model:cube, 
                        args): 
    """
    Computes the mutual inhibition loss for each model. 
    """

    shared_layer_list = []
    shared_layer_0 = torch.zeros(args.batch_size, 3, args.height, args.width).to(left_model.device)
    shared_layer_list.append(shared_layer_0)
    for t in range(args.num_timesteps-1): 
        y_left = left_model(shared_layer_list[-1], num_blocks=args.num_blocks_left, length=args.height)
        y_right = right_model(shared_layer_list[-1], num_blocks=args.num_blocks_right, length=args.height)

        left_model.Phi.append(left_model.Phi[-1] * (1-args.leak) + y_left * left_model.dt)
        left_model.Phi[-1][:, 0:3] *= 0
        left_model.Phi[-1][:, 0:3] += y_left[:, 0:3]

        right_model.Phi.append(right_model.Phi[-1] * (1-args.leak) + y_right * right_model.dt)
        right_model.Phi[-1][:, 0:3] *= 0
        right_model.Phi[-1][:, 0:3] += y_right[:, 0:3]


        shared_layer_next = (1-args.leak)*shared_layer_list[-1] + left_model.Phi[-1][:, 0:3] * args.dt + right_model.Phi[-1][:, 0:3] * args.dt
        shared_layer_list.append(shared_layer_next)

    loss = torch.mean((shared_layer_list[-1])**2)

    return loss, shared_layer_list




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

    log("Making left model...", log_path)
    left_model = cube(
        kernel_size = args.kernel_size,
        num_channels_list = args.num_channels_list, 
        block_overlap_depth = args.internal_block_overlap_depth,
        device = device, 
        leak = args.leak, 
        dt = args.dt, 
        sparsity_frac=args.left_obs_sparsity_frac
    ).to(device)
    log("Done making left model.", log_path)

    log("Making right model...", log_path)
    right_model = cube(
        kernel_size = args.kernel_size,
        num_channels_list = args.num_channels_list, 
        block_overlap_depth = args.internal_block_overlap_depth,
        device = device, 
        leak = args.leak, 
        dt = args.dt, 
        sparsity_frac=args.right_obs_sparsity_frac
    ).to(device)
    log("Done making right model.", log_path)


    # set up optimizer 
    left_optimizer = optim.Adam(left_model.parameters(), lr=args.lr_left)
    right_optimizer = optim.Adam(right_model.parameters(), lr=args.lr_right)

    left_model.train()

    # train model -- let's start with a test loss computation call
    loss, shared_layer_list = get_inhibition_loss(left_model, right_model, args)

    log(f"Initial loss: {loss.item()}", log_path)

    # optimize left model's weights
    for epoch in range(args.num_epochs): 
        left_optimizer.zero_grad()
        right_optimizer.zero_grad()

        loss, shared_layer_list = get_inhibition_loss(left_model, right_model, args)
        loss.backward()
        left_optimizer.step()

        left_model.clip_weights()

        log(f"Epoch {epoch}: Loss: {loss.item()}", log_path)
        if epoch % args.save_every == 0: 
            torch.save(left_model.state_dict(), os.path.join(args.save_dir, f"left_model_epoch{epoch}.pt"))
            torch.save(right_model.state_dict(), os.path.join(args.save_dir, f"right_model_epoch{epoch}.pt"))




if __name__ == "__main__": 
    main()