from moviepy.editor import VideoFileClip

# Define the path to your original video
source_video_path = '/home/xucao2/tool/SocialGesture/data_sources/In_Cabin/2024_0224_175858_I.mp4'

# Define the path for the resized video output
output_video_path = '/home/xucao2/tool/SocialGesture/data_sources/In_Cabin/2024_0224_175858_I_360p.mp4'

# Load the source video
clip = VideoFileClip(source_video_path)

# Resize the video. The height is set to 360 pixels. 
# The width will be calculated automatically to maintain the aspect ratio.
clip_resized = clip.resize(height=360)

# Write the resized video to the output file
clip_resized.write_videofile(output_video_path)