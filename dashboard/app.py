import streamlit as st
import os
import sys
import json
from pathlib import Path
import re
import subprocess
import pandas as pd
from datetime import datetime
import glob

sys.path.append("cortex_cube")
from val import eval_checkpoint_over_block_length_range



def is_file_open(file_path):
    """Check if a file is currently open (being written to)."""
    try:
        with open(file_path, 'r+') as f:
            return False
    except IOError:
        return True

def find_experiment_folders(root_dir='results'):
    """Find all experiment folders with required files and timestamps."""
    experiment_folders = []
    timestamp_pattern = r'\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_UTC'
    
    for path in Path(root_dir).rglob('*'):
        if path.is_dir() and re.search(timestamp_pattern, str(path)):
            args_file = path / 'args.json'
            loss_log = path / 'loss.log'
            grad_log = path / 'gradient_magnitudes.log'
            
            if args_file.exists() and loss_log.exists() and grad_log.exists():
                is_active = is_file_open(loss_log) or is_file_open(grad_log)
                experiment_folders.append((str(path), is_active))
    
    return experiment_folders

def get_loss_log(folder_path):
    """Get the entire contents of the loss.log file."""
    loss_log_path = os.path.join(folder_path, 'loss.log')
    print("Getting log from ", loss_log_path)
    try:
        with open(loss_log_path, 'r') as file:
            return file.read()
    except IOError:
        return "Error reading loss.log file"
    
def parse_loss_log(loss_log):
    """Parse the loss log and return a DataFrame with loss information."""
    pattern = r'\[(.*?)\] Epoch: (\d+) Loss: ([\d.]+) Batch: (\d+) Frames: (\d+)'
    entries = []

    for line in loss_log.split('\n'):
        match = re.search(pattern, line)
        if match:
            timestamp, epoch, loss, batch, frames = match.groups()
            entries.append({
                'timestamp': datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S'),
                'epoch': int(epoch),
                'loss': float(loss),
                'batch': int(batch),
                'frames': int(frames)
            })

    return pd.DataFrame(entries)

import streamlit as st
import pandas as pd
import os
import re
from datetime import datetime

def generate_overview_dataframe(experiment_folders):
    """
    Generate an overview dataframe for all experiment folders.

    This function reads the loss.log file for each experiment folder,
    extracts the final training loss and epoch, and the last validation loss.

    Args:
    experiment_folders (list): A list of tuples, each containing
                               (folder_path, is_active) for each experiment.

    Returns:
    pd.DataFrame: A dataframe with columns for folder name, final epoch,
                  final training loss, and last validation loss.
    """
    overview_data = []

    for folder_path, is_active in experiment_folders:
        folder_name = os.path.basename(folder_path)
        loss_log_path = os.path.join(folder_path, 'loss.log')

        try:
            with open(loss_log_path, 'r') as file:
                lines = file.readlines()

            # Initialize variables
            final_epoch = None
            final_train_loss = None
            last_val_loss = None

            # Regex patterns
            train_pattern = r'\[(.*?)\] Epoch: (\d+) Loss: ([\d.]+) Batch: (\d+) Frames: (\d+)'
            val_pattern = r'\[(.*?)\] Validation loss: ([\d.]+)'

            # Process lines in reverse order to get the latest information
            for line in reversed(lines):
                if final_epoch is None and final_train_loss is None:
                    train_match = re.search(train_pattern, line)
                    if train_match:
                        final_epoch = int(train_match.group(2))
                        final_train_loss = float(train_match.group(3))

                if last_val_loss is None:
                    val_match = re.search(val_pattern, line)
                    if val_match:
                        last_val_loss = float(val_match.group(2))

                if final_epoch is not None and final_train_loss is not None and last_val_loss is not None:
                    break

            overview_data.append({
                'Folder': folder_name,
                'Final Epoch': final_epoch,
                'Final Training Loss': final_train_loss,
                'Last Validation Loss': last_val_loss,
                'Active': is_active
            })

        except IOError:
            print(f"Error reading loss.log file for {folder_path}")

    return pd.DataFrame(overview_data)








def main():
    st.title("Experiment Monitor")

    experiment_folders = find_experiment_folders()

    if not experiment_folders:
        st.warning("No experiment folders found.")
    else:
        st.write(f"Found {len(experiment_folders)} experiment folders:")
        
        # Create options for dropdown
        options = []
        for folder, is_active in experiment_folders:
            label = f"{folder} {'(Active)' if is_active else ''}"
            options.append(label)
        
        # Dropdown for selecting experiment
        selected_experiment = st.selectbox("Select an experiment:", options)
        
        if selected_experiment:
            # Extract folder path from selected option
            selected_folder = selected_experiment.split(" (Active)")[0]
            
            # Display tail of loss.log
            # stript selected_folder for whitespace 

            selected_folder = selected_folder.strip()
            loss_log = get_loss_log(selected_folder)

            # st.subheader(f"Last 20 lines of loss.log of `\"{selected_folder}\"`:")
            # last_20_lines = '\n'.join(loss_log.split('\n')[-20:])
            # st.text(f"```\n{last_20_lines}\n```")

            # display as scrollable element
            st.write("## Loss Log")
            st.write(f"Selected folder: `\"{selected_folder}\"`")
            st.text_area("Loss Log", loss_log, height=400
            )

            # Parse the loss log and create a DataFrame
            df = parse_loss_log(loss_log)
            
            if not df.empty:
                # Plot loss over time
                st.subheader("Loss over time")
                st.line_chart(df.set_index('timestamp')['loss'])
                
                # Display the DataFrame
                st.subheader("Loss Data")
                st.dataframe(df)
            else:
                st.warning("No valid loss entries found in the log.")

    st.write("## Results Overview")
    # Add this to your main() function or wherever appropriate in your Streamlit app

    if st.button("Refresh Results Overview"):
        overview_df = generate_overview_dataframe(experiment_folders)
        st.dataframe(overview_df)


    if st.button("Generate validation report"):
        # get args.json from selected_experiment/args.json
        selected_folder = selected_experiment.split(" (Active)")[0].strip()
        args_path = os.path.join(selected_folder, 'args.json')
        with open(args_path, 'r') as f:
            args = json.load(f)
        # print to screen 
        st.write("## Validation Report")
        st.write("### Experiment Details")
        st.write(f"**Experiment Folder:** {selected_folder}")
        st.write(f"**Args path**: `\"{args_path}\"`")
        st.write(f"**Arguments:**")
        st.json(args)
        st.write(f"**Validation directory**: `\"{args['val_dir']}\"`")

        # glob the validation paths: 
        val_paths = glob.glob(f"{args['val_dir']}/*.mp4")
        st.write(f"**Validation paths:**")
        st.write(val_paths)

        # run validation script
        eval_df = eval_checkpoint_over_block_length_range(selected_folder, val_paths)
        st.write("### Evaluation Results")
        st.write(eval_df)

if __name__ == "__main__":
    main()