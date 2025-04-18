from moviepy.editor import VideoFileClip
import math
import os

# Load the video file
# video_path = '/mnt/data/Lets Play SHERIFF OF NOTTINGHAM!  Overboard Episode 9.mp4'

main_path = "/home/xucao2/tool/SocialGesture/data_sources/SocialGesture/Advent_Calendars"
out_path = "/home/xucao2/SocialGesture/data_sources/socialgesture/Advent_Calendars"

for v_path in os.listdir(main_path):
    
    video_path = os.path.join(main_path, v_path)
    video = VideoFileClip(video_path)

    # Video duration in seconds
    video_duration = int(video.duration)
    # Length of each clip in seconds (e.g., 2.5 minutes)
    clip_length = 5 * 60

    # Calculate the number of clips
    num_clips = math.ceil(video_duration / clip_length)

    # Loop through and create each clip
    for i in range(num_clips):
        start_time = i * clip_length
        end_time = min((i + 1) * clip_length, video_duration)
        
        # Remove video clips < 3min
        if end_time - start_time < 3 * 60:
            break
        
        # Create the clip
        clip = video.subclip(start_time, end_time)
        
        # Save the clip
        output_filename = f'clip_{i+1}_{start_time}_{end_time}' + '_' + v_path
        
        output_filename = os.path.join(out_path, output_filename)
        clip.write_videofile(output_filename, codec='libx264', audio_codec='aac')
        
        print(f'Created {output_filename}')
        
    video.close()
    
