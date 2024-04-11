import os
import bz2
import math
import argparse
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip

def compress_video(input_path, output_path):
    with open(input_path, 'rb') as f_in:
        with bz2.open(output_path, 'wb', compresslevel=9) as f_out:
            f_out.write(f_in.read())

def get_curves(path_to_mp4, output_dir='fractional_videos'):
    video = VideoFileClip(path_to_mp4)
    duration = math.floor(video.duration)
    print("Duration: ", duration)

    os.makedirs(output_dir, exist_ok=True)

    size_over_time = []
    for i in range(1, duration+1):
        fraction = i / duration
        end_time = duration * fraction
        fractional_video = video.subclip(0, end_time)
        output_name = f"{os.path.splitext(os.path.basename(path_to_mp4))[0]}_{i}_of_10th.mp4"
        output_path = os.path.join(output_dir, output_name)
        print(f"Writing {output_name} to {output_path}")
        fractional_video.write_videofile(output_path, fps=30)

        compressed_path = output_path + '.gz'
        compress_video(output_path, compressed_path)
        compressed_size = os.path.getsize(compressed_path)
        size_over_time.append(compressed_size)
        print(f"Compressed size of {output_name}: {compressed_size} bytes")

    video.close()
    return size_over_time

def main(args):
    k400_path = 'dataset/k400/train'
    mp4_files_ = os.listdir(k400_path)
    mp4_files = [os.path.join(k400_path, i) for i in mp4_files_]
    mp4_files.sort(reverse=True)

    sizes = []
    cache_dir = args.cache_dir
    os.makedirs(cache_dir, exist_ok=True)

    for i in range(args.num_videos_to_measure):
        if i % args.num_workers == args.worker_num:
            out_path = os.path.join(cache_dir, f'{i}.txt')
            # if already exists, skip
            if os.path.exists(out_path):
                print(f"{i}.txt Already exists, skipping...")
                continue
            print(f"\n\n=== {i} of {args.num_videos_to_measure} ===\n\n")
            try: 
                sizes.append(get_curves(mp4_files[i], output_dir='fractional_videos'))
                with open(out_path, 'w') as f:
                    out_str = f'{sizes[-1]}'
                    f.write(out_str)
            except Exception as e:
                print(f"Error: {e}\nContinuing...")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_videos_to_measure', type=int, default=200)
    parser.add_argument('--worker_num', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--cache_dir', type=str, default='results/cache/')
    args = parser.parse_args()

    main(args)