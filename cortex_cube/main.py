# Main training loop for the cube cortex video predictive coderl:
# run from root of repo `python3 cortex_cube/main.py`

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from cube_model import cube

import glob

from computronium.video_utils import async_video_loader, video_data_generator

import pdb


DATA_DIR = "dataset/debug/"


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


# glob all mp4 files in the directory
video_paths = glob.glob(DATA_DIR + "*.mp4")
num_videos = 3

for video in async_video_loader(video_paths[:3], num_workers=1, rescale=[64, 64]):
    print("video shape: ", video.shape)
    visualize_frames(video, num_frames=5, output_path='figures/debug_load.png')
    break


# use video_data_generator(video_paths, batch_size, 
                        #  num_workers=4, 
                        #  rescale=[240, 360], 
                        #  float01 = True):
# to load a batch of videos

for batch_data_np in video_data_generator(video_paths[:10], batch_size=2, num_workers=1, rescale=[64, 64], float01=True):
    print("batch_data_np shape: ", batch_data_np.shape)
    break


# Set up the model
NUM_EPOCHS = 10

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("mps:0")
model = cube(kernel_size=3, num_channels_list=[16, 128, 16], num_blocks=3, block_overlap_depth=1, device=device).to(device)

# Set up the optimizer
model_input_convs = [conv.weight for conv in model.input_conv]
model_body_convs = [conv.weight for conv in model.body_conv]
model_params = model_input_convs + model_body_convs
optimizer = optim.Adam(model_params, lr=0.001)

# Set up data. For now we will use dummy data.

num_timesteps = 100

# data = torch.tensor(batch_data_np, dtype=torch.float32).to(device)

pdb.set_trace()

losses = []
# for epoch in range(NUM_EPOCHS):
epoch = 0
for data_np in video_data_generator(video_paths[:10], batch_size=2, num_workers=1, rescale=[64, 64], float01=True): 

    data = torch.tensor(data_np, dtype=torch.float32).to(device)
    num_timesteps = data.shape[0]

    optimizer.zero_grad()
    loss = 0
    for t in range(num_timesteps-1):
        x = data[t]
        y = model(x)
        loss += model.loss(y[:, 0:3], data[t+1])
    
    loss.backward()
    optimizer.step()
    losses += [loss.item()]

    epoch += 1
    print("Epoch: {}, Loss: {}".format(epoch, loss.item()))
    if epoch > NUM_EPOCHS:
        break


# Plot the loss over time
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()
# save to figures 
plt.savefig('figures/debug_loss.png')


