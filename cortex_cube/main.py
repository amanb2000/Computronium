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

from computronium.video_utils import async_video_loader, video_data_generator

import pdb



torch.autograd.set_detect_anomaly(True)

DATA_DIR = "dataset/debug/"
OUT_DIR = "results/debug/"
BATCH_SIZE = 5
LR = 0.001
NUM_EPOCHS = 1000
NUM_WORKERS = 5

video_paths = glob.glob(DATA_DIR + "*.mp4")
num_videos = 5

# if it doesn't exist, make OUT_DIR
if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)


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



# use video_data_generator(video_paths, batch_size, 
                        #  num_workers=4, 
                        #  rescale=[240, 360], 
                        #  float01 = True):
# to load a batch of videos

for batch_data_np in video_data_generator(video_paths[:10], batch_size=2, num_workers=1, rescale=[64, 64], float01=True):
    print("batch_data_np shape: ", batch_data_np.shape)
    break



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("mps:0")
model = cube(kernel_size=3, num_channels_list=[16, 128, 16], num_blocks=3, block_overlap_depth=1, device=device).to(device)

# Set up the optimizer
model_input_convs = [conv.weight for conv in model.input_conv]
model_body_convs = [conv.weight for conv in model.body_conv]
model_params = model_input_convs + model_body_convs
optimizer = optim.Adam(model_params, lr=LR)

# print out the model shape and number of parameters really pretty
print(model)
print("Number of parameters: ", sum(p.numel() for p in model_params if p.requires_grad))


# Set up data. For now we will use dummy data.
# data = torch.tensor(batch_data_np, dtype=torch.float32).to(device)

losses = []
# for epoch in range(NUM_EPOCHS):
epoch = 0
for data_np in video_data_generator(video_paths[:num_videos], batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, rescale=[64, 64], float01=True): 
    data = torch.tensor(data_np, dtype=torch.float32).to(device)
    num_timesteps = data.shape[0]

    optimizer.zero_grad()
    loss = 0
    for t in range(num_timesteps-1):
        x = data[t]
        y = model(x)
        loss += model.loss(y[:, 0:3], data[t+1]) / num_timesteps

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
    if epoch > NUM_EPOCHS:
        break




# Plot the loss over time
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()
# save to figures 
plt.savefig(os.path.join(OUT_DIR, 'debug_loss.png'))





