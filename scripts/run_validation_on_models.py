import re
from pathlib import Path
from typing import List
import os, json, glob, sys

sys.path.append("cortex_cube")
from val import eval_checkpoint_over_block_length_range


def is_file_open(file_path: Path) -> bool:
    """Check if a file is open by another process."""
    try:
        with open(file_path, 'r') as file:
            return False
    except PermissionError:
        return True

def find_experiment_folders(root_dir='results'):
    """Find all experiment folders with required files and timestamps."""
    experiment_folders = []
    timestamp_pattern = r'\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_UTC'
    
    for path in Path(root_dir).rglob('*'):
        if path.is_dir():
            args_file = path / 'args.json'
            loss_log = path / 'loss.log'
            grad_log = path / 'gradient_magnitudes.log'
            model_file = path / 'best_model.pth'
            
            if args_file.exists() and loss_log.exists() and grad_log.exists() and model_file.exists():
                is_active = is_file_open(loss_log) or is_file_open(grad_log)
                experiment_folders.append((str(path), is_active))
    
    return experiment_folders



for selected_folder, active in find_experiment_folders():
    if active:
        continue

    print(selected_folder)
    args_path = os.path.join(selected_folder, 'args.json')
    with open(args_path, 'r') as f:
        args = json.load(f)
    val_paths = glob.glob(f"{args['val_dir']}/*.mp4")
    print(val_paths)
    
    # run validation script
    eval_df = eval_checkpoint_over_block_length_range(selected_folder, val_paths)