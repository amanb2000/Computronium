import torch
import json
from tqdm import tqdm
from main import get_loss_on_video_batch
from cube_model import cube
import os
from computronium.video_utils import async_video_loader, video_data_generator


class DictToObject:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            setattr(self, key, value)


# load a pretrained model
def initialize_pretrained_model(model_path, args_path, device):

    # load args from json file
    with open(args_path, 'r') as f:
        args = DictToObject(json.load(f))
    
    # load the model
    model = cube(
        kernel_size=args.kernel_size, 
        num_channels_list=args.num_channels_list, 
        block_overlap_depth=args.block_overlap_depth,
        device=device,
        leak=args.leak,
        dt=args.dt,
        sparsity_frac=args.sparsity_frac).to(device)
    model.load_model(model_path)

    return model, args

# evaluate a model checkpoint on a validation_loader
@ torch.no_grad()
def eval_checkpoint(model, args, val_loader, num_iters, num_blocks=None, length=None, loss_label="MSE"):
    total_val_loss = 0
    cnt = 0
    for data_np_val in tqdm(val_loader):
        model.Phi = []
        model.correlation_tensor = []
        val_loss = get_loss_on_video_batch(data_np_val, 
                                           args,
                                           model, 
                                           args.num_steps_per_frame, 
                                           args.weight_regularization, 
                                           args.activity_regularization, 
                                           num_blocks=num_blocks, 
                                           length=length, 
                                           loss_label=loss_label,
                                           zero_grad=False)
        total_val_loss += val_loss.item()
        cnt += 1

        if cnt > num_iters: 
            break
    
    mean_val_loss = total_val_loss / cnt
    return mean_val_loss

import pandas as pd
from itertools import product
# block_range_iter and length_range_iter are iterables of the range of values to test
def eval_checkpoint_over_block_length_range(dir_path, val_paths, block_range_iter=None, length_range_iter=None, loss_label="MSE", device="cuda:0"):
    # save results to a dataframe

    # if file exists, skip
    if os.path.exists(os.path.join(dir_path, "val_block_length_results.csv")):
        print("Loading existing results")
        return pd.read_csv(os.path.join(dir_path, "val_block_length_results.csv"))

    model_path = os.path.join(dir_path, "best_model.pth")
    args_path = os.path.join(dir_path, "args.json")
    model, args = initialize_pretrained_model(model_path, args_path, device)

    if block_range_iter is None:
        block_range_iter = range(args.min_num_blocks, 2 * args.max_num_blocks, 2)

    if length_range_iter is None:
        length_range_iter = range(args.min_length, 2 * args.max_length, 10)

    val_loader = video_data_generator(val_paths, batch_size=args.batch_size, num_workers=args.num_workers, rescale=[args.video_height, args.video_width], float01=True)

    num_iters = (len(val_paths) // args.batch_size) + 1


    data = {"num_blocks": [], "length": [], "mean_val_loss": []}

    for num_blocks, length in product(block_range_iter, length_range_iter):
        mean_val_loss = eval_checkpoint(model, args, val_loader, num_iters, num_blocks=num_blocks, length=length, loss_label=loss_label)
        data["num_blocks"].append(num_blocks)
        data["length"].append(length)
        data["mean_val_loss"].append(mean_val_loss)
    
    results_df = pd.DataFrame(data)
    results_df.to_csv(os.path.join(dir_path, "val_block_length_results.csv"))

    return results_df
