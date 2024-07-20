import os
import subprocess

# Path to the directory containing the videos
input_directory = '/home/xucao2/SocialGesture/data_sources/500_video'

# Path to the directory where you want to save the transformed videos
output_directory = '/home/xucao2/SocialGesture/data_sources/500_video_5fps'

# Ensure the output directory exists
os.makedirs(output_directory, exist_ok=True)

# Loop through all files in the input directory
for file in os.listdir(input_directory):
    if file.endswith(('.mp4', '.mkv', '.avi', '.mov')):  # Add other video formats if needed
        input_path = os.path.join(input_directory, file)
        output_path = os.path.join(output_directory, file)
        
        # Command to change the frame rate to 5 fps
        command = [
            'ffmpeg',
            '-i', input_path,
            '-filter:v', 'fps=fps=5',
            output_path
        ]
        
        # Execute the command
        subprocess.run(command, check=True)

print(f"Successfully transformed all videos to 5 fps and saved them to {output_directory}")