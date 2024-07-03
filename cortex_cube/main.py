# Main training loop for the cube cortex video predictive coderl:
# run from root of repo `python3 cortex_cube/main.py`

import os
import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt
from cube_model import cube

import glob
import json

from computronium.video_utils import async_video_loader, video_data_generator

import pdb
import argparse 

# TODO: Make commandline args with defaults as below. 
# edit the DATA_DIR and the other constants as DATA_DIR = args.data_dir, etc.
def parse_args():
    parser = argparse.ArgumentParser(description="Cube Cortex Video Predictive Coder")
    parser.add_argument('--data_dir', type=str, default="dataset/debug/", help="Directory containing the dataset. Default=dataset/debug/")
    parser.add_argument('--out_dir', type=str, default="results/debug/", help="Directory to save results. Default=results/debug/")
    parser.add_argument('--batch_size', type=int, default=5, help="Batch size. Default=5")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate. Default=0.001")
    parser.add_argument('--num_epochs', type=int, default=1000, help="Number of epochs. Default=1000")
    parser.add_argument('--num_workers', type=int, default=5, help="Number of workers for data loading. Default=5")
    parser.add_argument('--video_height', type=int, default=64, help="Video height. Default=64")
    parser.add_argument('--video_width', type=int, default=64, help="Video width. Default=64")
    parser.add_argument('--kernel_size', type=int, default=3, help="Kernel size. Defualt=3")
    parser.add_argument('--num_channels_list', nargs='+', type=int, default=[16, 128, 16], help="List of channel numbers. Defualt=[16,128,16]")
    parser.add_argument('--num_blocks', type=int, default=3, help="Number of blocks. Default=3")
    parser.add_argument('--block_overlap_depth', type=int, default=1, help="Block overlap depth. Default=1")
    parser.add_argument('--weight_regularization', type=float, default=1.0, help="Weight regularization. Default=1.0")
    parser.add_argument('--activity_regularization', type=float, default=1.0, help="Activity regularization. Defualt=1.0")
    parser.add_argument('--mps', action="store_true", help="Include to use mps accelerator. Default=use cuda if available, CPU if not")
    parser.add_argument('--num_overfit_videos', type=int, default=-1, help="How many videos to use to overfit the model. Default=-1 (use all data in folder)")
    return parser.parse_args()

args = parse_args()

# Use the parsed arguments
DATA_DIR = args.data_dir
OUT_DIR = args.out_dir
BATCH_SIZE = args.batch_size
LR = args.lr
NUM_EPOCHS = args.num_epochs
NUM_WORKERS = args.num_workers
VIDEO_HEIGHT = args.video_height
VIDEO_WIDTH = args.video_width
KERNEL_SIZE = args.kernel_size
NUM_CHANNELS_LIST = args.num_channels_list
NUM_BLOCKS = args.num_blocks
BLOCK_OVERLAP_DEPTH = args.block_overlap_depth
WEIGHT_REGULARIZATION = args.weight_regularization
ACTIVITY_REGULARIZATION = args.activity_regularization




video_paths = glob.glob(DATA_DIR + "*.mp4")
if args.num_overfit_videos > 0: 
    video_paths = video_paths[:args.num_overfit_videos]
print("Length of video paths: ", len(video_paths))

model_output_path = os.path.join(OUT_DIR, 'best_model.pth')

# if it doesn't exist, make OUT_DIR
if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)

# save the args as json in OUT_DIR/args.json
with open(os.path.join(OUT_DIR, 'args.json'), 'w') as f:
    json.dump(vars(args), f)



def visualize_frames(video, num_frames=5, output_path='figures/debug_load.png'):
    total_frames = video.shape[0]
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    fig, axes = plt.subplots(1, num_frames, figsize=(20, 4))
    fig.suptitle('Equally-spaced frames from video')
    
    for i, idx in enumerate(indices):
        axes[i].imshow(video[idx])
        axes[i].axis('off')
        axes[i].set_title(f'Frame {idx}')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


if not args.mps: 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
else: 
    device = torch.device("mps:0")


model = cube(kernel_size=KERNEL_SIZE, num_channels_list=NUM_CHANNELS_LIST, num_blocks=NUM_BLOCKS, block_overlap_depth=BLOCK_OVERLAP_DEPTH, device=device).to(device)

# Set up the optimizer
optimizer = optim.Adam(model.parameters(), lr=LR)

# print out the model shape and number of parameters really pretty
print(model)
print("Number of parameters: ", sum(p.numel() for p in model.parameters()))


# Set up data. For now we will use dummy data.
# data = torch.tensor(batch_data_np, dtype=torch.float32).to(device)

losses = []
# for epoch in range(NUM_EPOCHS):
epoch = 0
best_loss = 1000000
for data_np in video_data_generator(video_paths, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, rescale=[VIDEO_HEIGHT, VIDEO_WIDTH], float01=True): 
    data = torch.tensor(data_np, dtype=torch.float32).to(device)
    num_timesteps = data.shape[0]

    optimizer.zero_grad()
    loss = 0
    for t in range(num_timesteps-1):
        x = data[t]
        y = model(x)
        loss += model.loss(y[:, 0:3], data[t+1], weight_regularization=WEIGHT_REGULARIZATION, activation_regularization=ACTIVITY_REGULARIZATION) / num_timesteps

        model.Phi.append(model.Phi[-1] + y * model.dt)
        # save the instrumentation values 
        # inst_value = model.instrumentation_values()
        # save to OUT_DIR/instrumentation_values.txt
        # with open(OUT_DIR + f'instrumentation_values_ep{epoch}_frame{t}.txt', 'w') as f:
            # f.write(inst_value)
    
    loss.backward()
    optimizer.step()
    losses += [loss.item()]

    model.Phi = []

    epoch += 1
    print("Epoch: {}, Loss: {}, Batch: {}, Frames: {}".format(epoch, loss.item(), data.shape[1], data.shape[0]))
    if loss.item() < best_loss:
        print("Saving best model weights at ", model_output_path, "...")
        best_loss = loss.item()
        model.save_model(model_output_path)
        print("Model saved.\n")
        # todo: add validation set, etc. 

    if epoch > NUM_EPOCHS:
        break




# Plot the loss over time
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()
# save to figures 
plt.savefig(os.path.join(OUT_DIR, 'debug_loss.png'))
