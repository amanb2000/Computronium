# Import box
import os
import sys
import jax 
import jax.numpy as jnp


import numpy as np
import cv2
import av 
import concurrent.futures
import pydantic 
from tqdm import tqdm

import pdb


def rescale_frame(frame, height_width): 
    """
    frame: numpy array of shape [height, width, channel]
    height_width: tuple of (height, width)
    """
    # what is the scale factor for height and width? 
    hf, wf = frame.shape[:2] # height fo frame, width of frame
    hd, wd = height_width # height of desired, width of desired

    # print("hf, wf, hd, wd", hf, wf, hd, wd)

    # if hf/hd > wf/wd: 
    if hd / hf < wd / wf:
        # scale by width 
        scale = wd/wf
        frame_ = cv2.resize(frame, (round(wf*scale), round(hf*scale)))
        # print("Scaled frame size: ", frame_.shape)
        # crop to height 
        h = frame_.shape[0]
        frame_ = frame_[(h-hd)//2:(h+hd)//2, :]
        # print("Cropped frame size: ", frame_.shape)
        
    else:
        # scale by height 
        scale = hd/hf
        frame_ = cv2.resize(frame, (round(wf*scale), round(hf*scale)))
        # crop to width  
        w = frame_.shape[1]
        frame_ = frame_[:, (w-wd)//2:(w+wd)//2]
    return frame_

def load_video(video_path):
    """ Function to load a video from a given path using pyav.

    Returns numpy array of shape [num_frames, height, width, channels]
    """
    try: 
        with av.open(video_path) as container:
            frames = []
            for frame in container.decode(video=0):
                frame = frame.to_ndarray(format='rgb24')
                frames.append(frame)
        return np.stack(frames)
    except Exception as e: 
        print(f"[load_video] failed to load video at {video_path} with exception {e}")
        return None

def async_video_loader(video_paths, num_workers, rescale=[240, 360]):
    """ Given a set of video paths, number of workers, and a rescale factor, 
    this function returns a numpy stack of video data of shape [num_videos,
    num_frames, height, width, channels] using a ThreadPoolExecutor with
    num_workers to load the videos in parallel with pyav. 

    The videos are rescaled to the desired height and width, and cropped to the
    minimum length of all videos in the batch.
    """
    with concurrent.futures.ThreadPoolExecutor(num_workers) as executor:
        futures = [executor.submit(load_video, path) for path in video_paths]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
    # discard any results that are -1
    results = [result for result in results if result is not None]
    # check for shortest video 
    # print("Cropping videos in batch to min size...")
    # print("Type of results: ", type(results))
    # print("Type of results[0]: ", type(results[0]))
    for i in range(len(results)):
        # reshape to rescale
        # results[i] = np.array([cv2.resize(frame, (rescale[1], rescale[0])) for frame in results[i]])
        # crop to size 
        results[i] = np.array([rescale_frame(frame, rescale) for frame in results[i]])
        
        # print("Shape of results[{i}]: ", results[i].shape)

    min_length = min(video.shape[0] for video in results)

    while min_length == 0: 
        # remove videos with 0 length
        print("Loaded 0 length video, removing...")
        results = [video for video in results if video.shape[0] > 0]
        min_length = min(video.shape[0] for video in results)
    # print("min length: ", min_length)
    results = [video[:min_length] for video in results]
    try: 
        stacked_results = np.stack(results)
    except: 
        pdb.set_trace()
    return stacked_results

def video_data_generator(video_paths, batch_size, 
                         num_workers=4, 
                         rescale=[240, 360], 
                         float01 = True):
    """ Given a list of mp4 video paths, a batch size, num_workers, and a video rescale size, 
    this provides an asynchronous multi-threaded generator that yields batches of video data 
    as jax.numpy arrays of shape [num_frames, batch_size, channels, height, width].
    """
    num_videos = len(video_paths)
    # for i in range(0, num_videos, batch_size):
    i = 0
    while True: 
        batch_paths = video_paths[i:i+batch_size]
        # batch_data = [load_video(path) for path in batch_paths]
        batch_data = async_video_loader(batch_paths, num_workers=num_workers, rescale=rescale)
        # comes ouot as [batch_size, num_frames, height, width, channels]
        # reorg to [num_frames, batch_size, channels, height, width]
        batch_data = np.transpose(batch_data, (1, 0, 4, 2, 3))
        # convert to float, divide by 255
        if float01:
            batch_data = batch_data.astype(np.float32) / 255.0
        yield np.array(batch_data)

        i += batch_size 
        if i > num_videos - batch_size:
            i = 0


