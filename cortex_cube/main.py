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
from visualizer import create_phi_batch_list, save_video_from_phi_list

import pdb
import argparse 
from datetime import datetime
from scipy.stats import pearsonr

from tqdm import tqdm 

def log(msg, file_path): 
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(file_path, 'a') as f: 
        f.write(f"[{current_time}] {msg}\n")


# TODO: Make commandline args with defaults as below. 
# edit the DATA_DIR and the other constants as DATA_DIR = args.data_dir, etc.
def parse_args():
    parser = argparse.ArgumentParser(description="Cube Cortex Video Predictive Coder")
    parser.add_argument('--data_dir', type=str, default="dataset/debug/", help="Directory containing the dataset. Default=dataset/debug/")
    parser.add_argument('--val_dir', type=str, default="dataset/debug/", help="Directory containing the validation dataset. Default=dataset/debug/")
    parser.add_argument('--val_period', type=int, default=10, help="How many batches between each validation run? Default=10")

    parser.add_argument('--out_dir', type=str, default="results/debug/", help="Directory to save results. Default=results/debug/")
    parser.add_argument('--batch_size', type=int, default=5, help="Batch size. Default=5")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate. Default=0.001")
    parser.add_argument('--num_epochs', type=int, default=1000, help="Number of epochs. Default=1000")
    parser.add_argument('--num_workers', type=int, default=5, help="Number of workers for data loading. Default=5")
    parser.add_argument('--video_height', type=int, default=64, help="Video height. Default=64")
    parser.add_argument('--video_width', type=int, default=64, help="Video width. Default=64")

    parser.add_argument('--kernel_size', type=int, default=3, help="Kernel size. Defualt=3")
    parser.add_argument('--num_channels_list', nargs='+', type=int, default=[16, 128, 16], help="List of channel numbers. Defualt=[16,128,16]")

    parser.add_argument('--min_num_blocks', type=int, default=2, help="Minimum number of blocks. Default=2")
    parser.add_argument('--max_num_blocks', type=int, default=5, help="Maximum number of blocks. Default=5")

    parser.add_argument('--min_length', type=int, default=64, help="Minimum extra length on Phi tensor. Default=64")
    parser.add_argument('--max_length', type=int, default=128, help="Maximum extra length on Phi tensor. Default=128")

    parser.add_argument('--block_overlap_depth', type=int, default=1, help="Block overlap depth. Default=1")
    parser.add_argument('--weight_regularization', type=float, default=1.0, help="Weight regularization. Default=1.0")
    parser.add_argument('--activity_regularization', type=float, default=1.0, help="Activity regularization. Defualt=1.0")
    parser.add_argument('--sparsity_frac', type=float, default=0.99, help="Sparsity fraction for frames input to model. Default=0.99")
    parser.add_argument('--mps', action="store_true", help="Include to use mps accelerator. Default=use cuda if available, CPU if not")
    parser.add_argument('--leak', type=float, default=0.1, help="Leak value for leaky state tensor, Phi *= (1-leak). Default=0, max=1")
    parser.add_argument('--dt', type=float, default=0.1, help="Time step for the model. Default=0.1")
    parser.add_argument('--num_steps_per_frame', type=int, default=1, help="Number of steps per frame. Default=1")

    parser.add_argument('--num_overfit_videos', type=int, default=-1, help="How many videos to use to overfit the model. Default=-1 (use all data in folder)")
    parser.add_argument('--visualization_period', type=int, default=10, help="How training steps between each video saved to disk? Default=10")

    return parser.parse_args()


args = parse_args()
assert args.min_length >= args.video_width


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
MIN_NUM_BLOCKS = args.min_num_blocks
MAX_NUM_BLOCKS = args.max_num_blocks
MIN_LENGTH = args.min_length
MAX_LENGTH = args.max_length
BLOCK_OVERLAP_DEPTH = args.block_overlap_depth
WEIGHT_REGULARIZATION = args.weight_regularization
ACTIVITY_REGULARIZATION = args.activity_regularization

# make the out_dir if it doesn't already exist 
if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)


# get the set of validation set paths 
val_paths = glob.glob(os.path.join(args.val_dir, "*.mp4"))

video_paths = glob.glob(os.path.join(DATA_DIR, "*.mp4"))
if args.num_overfit_videos > 0: 
    video_paths = video_paths[:args.num_overfit_videos]
    val_paths = val_paths[:args.num_overfit_videos]
    print("Debug video paths: ", video_paths)
print("Length of video paths: ", len(video_paths))
log("Length of video paths: " + str(len(video_paths)), os.path.join(OUT_DIR, 'loss.log'))

print("Length of validation paths: ", len(val_paths))
log("Length of validation paths: " + str(len(val_paths)), os.path.join(OUT_DIR, 'loss.log'))


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
print("Device: ", device)
log("Device: " + str(device), os.path.join(OUT_DIR, 'loss.log'))

model = cube(
    kernel_size=KERNEL_SIZE, 
    num_channels_list=NUM_CHANNELS_LIST, 
    block_overlap_depth=BLOCK_OVERLAP_DEPTH,
    device=device,
    leak=args.leak,
    dt=args.dt,
    sparsity_frac=args.sparsity_frac).to(device)

# Set up the optimizer
optimizer = optim.Adam(model.parameters(), lr=LR)

# print out the model shape and number of parameters really pretty
print(model)
print("Number of parameters: ", sum(p.numel() for p in model.parameters()))


# Function to call model on a numpy array of video frames, update the Phi tensor list 
# for a certain 
def get_loss_on_video_batch(data_np: np.ndarray, 
                            model: cube, 
                            num_steps_per_frame: int, 
                            weight_regularization: float, 
                            activity_regularization: float, 
                            instrument_correlations=True,
                            ):
    """
    Computes the loss of the model on a given dataset of videos 
    data_np of shape [frames, batch, rgb, width, height]
    Notes: 
        Zero the optimizer gradient beforehand 
        Use with torch.no_grad() to avoid grad comp on the val set
    Args: 
    """
    data = torch.tensor(data_np, dtype=torch.float32).to(device)
    num_timesteps = data.shape[0]

    num_blocks = np.random.randint(MIN_NUM_BLOCKS, MAX_NUM_BLOCKS + 1)
    length = np.random.randint(MIN_LENGTH, MAX_LENGTH + 1)
    print("Num blocks: ", num_blocks)
    print("Length: ", length)


    optimizer.zero_grad()
    loss = 0
    MSE_total = 0.0
    weight_reg_total = 0.0
    activity_reg_total = 0.0
    for t in range(num_timesteps-1):
        x = data[t]

        for subtimestep in range(num_steps_per_frame): 
            y = model(x, instrument_correlations=instrument_correlations, num_blocks=num_blocks, length=length)

            model.Phi.append(model.Phi[-1]*(1-args.leak) + y * model.dt)
            model.Phi[-1][:, 0:3] *= 0 # zero out the first 3 channels in the prediction layer
            model.Phi[-1][:, 0:3] += y[:, 0:3] # set the first 3 channels to the absolute prediction (non-differential)

        total_loss, MSE, weight_reg_loss, activity_reg_loss = model.loss(data[t+1],
                           y[:, 0:3], 
                           weight_regularization=weight_regularization, 
                           activation_regularization=activity_regularization)
        MSE_total += MSE 
        weight_reg_total += weight_reg_loss
        activity_reg_total += activity_reg_loss
    loss = (MSE_total + weight_reg_total + activity_reg_total) / num_timesteps 
    print("=== LOSS REPORT ===")
    print("\tMean loss over all frames in batch: ", loss)
    print("\tMean MSE over all frames: ", MSE_total / num_timesteps)
    print("\tMean weight reg: ", weight_reg_total / num_timesteps)
    print("\tMean activity reg: ", activity_reg_total / num_timesteps)

        # print("predicted mean: ", y[:, 0:3].mean())
        # print("real mean: ", data.mean())
        # save the instrumentation values 
        # inst_value = model.instrumentation_values()
        # save to OUT_DIR/instrumentation_values.txt
        # with open(OUT_DIR + f'instrumentation_values_ep{epoch}_frame{t}.txt', 'w') as f:
            # f.write(inst_value)
    return loss

def compute_correlation_tensor(model, epoch):
    correlation_tensor_actuallyatensornow = torch.cat([a[None, :] for a in model.correlation_tensor])
    # Shape is (timesteps, N, num_blocks-1, block_overlap_depth, m, n, 2)

    # Flatten the correlation tensor to be (num_blocks-1, timesteps*Nblock_overlap_depth*m*n, 2)
    # First permute
    cor_tensor_permuted = correlation_tensor_actuallyatensornow.permute(2, 0, 1, 3, 4, 5, 6)

    # Then reshape. Final shape is (num_blocks-1, timesteps*Nblock_overlap_depth*m*n, 2)
    cor_tensor_flattened = cor_tensor_permuted.reshape(cor_tensor_permuted.shape[0], -1, 2).detach().cpu().numpy()

    # Compute how correlated cor_tensor_flattened[0] is with cor_tensor_flattened[1] (pearson correlation coefficient)
    correlations = []
    for i in tqdm(range(cor_tensor_flattened.shape[0])):
        cor = pearsonr(cor_tensor_flattened[i, :, 0], cor_tensor_flattened[i, :, 1])
        
        correlations.append(cor.statistic)

    print("Correlations of layer overlap, measuring mutual inhibition.")
    for i, cor in enumerate(correlations):
        print(f"Block Junction {i}: {cor}")

    # Plot a random 1% of the correlations, with trendline
    plt.scatter(cor_tensor_flattened[i, :, 0], cor_tensor_flattened[i, :, 1], s=0.005, alpha=0.5)
    
    # save figure
    plt.savefig(os.path.join(OUT_DIR, f'correlation_scatter_ep{epoch}.png'))
    # clear figure for next time 
    plt.clf()

def log_grad_magnitudes(model, OUT_DIR): 
    # get gradient magnitudes
    grad_magnitudes = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_magnitudes[name] = param.grad.data.norm(2).item()
    
    # Log gradient magnitudes
    log_message = f"Epoch {epoch}: Gradient Magnitudes - {grad_magnitudes}"
    log(log_message, os.path.join(OUT_DIR, 'gradient_magnitudes.log'))




losses = []
val_losses = []
# for epoch in range(NUM_EPOCHS):
epoch = 0
best_loss = 1000000
for data_np in video_data_generator(video_paths, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, rescale=[VIDEO_HEIGHT, VIDEO_WIDTH], float01=True): 
    # get the loss on data_np of shape [timesteps, batch_size, height, width, channels]
    log("Starting get_loss_on_video_batch...", os.path.join(OUT_DIR, 'loss.log'))
    loss = get_loss_on_video_batch(data_np, model, args.num_steps_per_frame, WEIGHT_REGULARIZATION, ACTIVITY_REGULARIZATION, instrument_correlations=False) 
    log("Done get_loss_on_video_batch.", os.path.join(OUT_DIR, 'loss.log'))

    log("Performing gradient step...", os.path.join(OUT_DIR, 'loss.log'))
    loss.backward()
    optimizer.step()
    log("Done gradient step.", os.path.join(OUT_DIR, 'loss.log'))

    # compute the correlation tensor
    # log("Starting computation of correlation tensor...", os.path.join(OUT_DIR, 'loss.log'))
    # compute_correlation_tensor(model, epoch) 
    # log("Done computing correlation tensors.", os.path.join(OUT_DIR, 'loss.log'))

    log("Logging gradient magnitudes...", os.path.join(OUT_DIR, 'loss.log'))
    log_grad_magnitudes(model, OUT_DIR) 
    log("Done logging gradient magnitudes.", os.path.join(OUT_DIR, 'loss.log'))

    losses += [loss.item()]


    lg_str = "Epoch: {} Loss: {} Batch: {} Frames: {}".format(epoch, loss.item(), data_np.shape[1], data_np.shape[0])
    print(lg_str)
    log(lg_str, os.path.join(OUT_DIR, 'loss.log'))

    if epoch % args.visualization_period == 0: 
            log("Starting visualization...", os.path.join(OUT_DIR, 'loss.log'))
            vis_path = os.path.join(OUT_DIR, f"vis_ep{epoch}.mp4")
            print("Saving visualization to ", vis_path)
            batch_0_tmp_ = create_phi_batch_list(model.Phi) # take the 0th batch

            for k in range(len(batch_0_tmp_)):
                vis_path = os.path.join(OUT_DIR, f"vis_ep{epoch}_batchel{k}.mp4")
                batch_0_tmp = batch_0_tmp_[k]
                save_video_from_phi_list(batch_0_tmp, vis_path, width=VIDEO_WIDTH, height=VIDEO_HEIGHT, pixelnorm='clip')

                vis_path = os.path.join(OUT_DIR, f"baseline{epoch}_batchel{k}.mp4")
                # save_video_from_phi_list([i for i in torch.tensor(data_np)[:, k]], vis_path)
                save_video_from_phi_list([i for i in torch.tensor(data_np)[:, k]], vis_path, width=VIDEO_WIDTH, height=VIDEO_HEIGHT, pixelnorm='clip')


            print("Done!")
            log("Done visualizing.", os.path.join(OUT_DIR, 'loss.log'))

    # TODO: add validation set, etc. 
    # compute validation loss
    if epoch % args.val_period == 0:
        log("Computing validation loss...", os.path.join(OUT_DIR, 'loss.log'))
        with torch.no_grad(): 
            total_val_loss = 0
            cnt = 0
            for data_np_val in tqdm(video_data_generator(val_paths, 
                                                    batch_size=BATCH_SIZE, 
                                                    num_workers=NUM_WORKERS, 
                                                    rescale=[VIDEO_HEIGHT, VIDEO_WIDTH], 
                                                    float01=True)):
                model.Phi = []
                model.correlation_tensor = []
                val_loss = get_loss_on_video_batch(data_np_val, model, args.num_steps_per_frame, WEIGHT_REGULARIZATION, ACTIVITY_REGULARIZATION)
                total_val_loss += val_loss.item()
                cnt += 1

                if cnt > (len(val_paths) // BATCH_SIZE): 
                    break
        log('Done computing validation loss.', os.path.join(OUT_DIR, 'loss.log'))
        
        # compute the correlation tensor
        log("Starting computation of correlation tensor...", os.path.join(OUT_DIR, 'loss.log'))
        compute_correlation_tensor(model, epoch) 
        log("Done computing correlation tensors.", os.path.join(OUT_DIR, 'loss.log'))

        mean_val_loss = total_val_loss / cnt
        val_losses.append(mean_val_loss)
        log("Validation loss: " + str(mean_val_loss), os.path.join(OUT_DIR, 'loss.log'))


        if mean_val_loss < best_loss:
            print("Saving best model weights at ", model_output_path, "...")
            log("Saving best model weights at " + model_output_path + "...", os.path.join(OUT_DIR, 'loss.log'))
            best_loss = mean_val_loss 
            model.save_model(model_output_path)
            print("Model saved.\n")
            log("Model saved.", os.path.join(OUT_DIR, 'loss.log'))


    

    model.Phi = []
    model.correlation_tensor = []
    epoch += 1

    if epoch > NUM_EPOCHS:
        break




# Plot the loss over time
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()
# save to figures 
plt.savefig(os.path.join(OUT_DIR, 'debug_loss.png'))
