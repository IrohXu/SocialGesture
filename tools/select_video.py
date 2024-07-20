import os
import random
import shutil

# Path to the main directory containing subfolders with videos
main_directory = '/home/xucao2/SocialGesture/data_sources/socialgesture'

# Path to the directory where you want to copy the selected videos
output_directory = '/home/xucao2/SocialGesture/data_sources/500_video'

# Ensure the output directory exists
os.makedirs(output_directory, exist_ok=True)

# Collect all video file paths
video_files = []
for subdir in os.listdir(main_directory):
    subdir_path = os.path.join(main_directory, subdir)
    if os.path.isdir(subdir_path):
        for file in os.listdir(subdir_path):
            if file.endswith(('.mp4', '.mkv', '.avi', '.mov')):  # Add other video formats if needed
                video_files.append(os.path.join(subdir_path, file))

# Check if there are enough videos
if len(video_files) < 500:
    raise ValueError("Not enough video files in the directory to select 500 videos.")

# Randomly select 500 videos
selected_videos = random.sample(video_files, 500)

# Copy selected videos to the output directory
for video in selected_videos:
    shutil.copy(video, output_directory)

print(f"Successfully copied 500 random videos to {output_directory}")