import os
from moviepy.editor import ImageSequenceClip

def create_video_from_images(visualization_path, output_path, frame_rate=5):
    # Get list of images in the folder
    
    os.makedirs(output_path, exist_ok=True)
    
    for image_folder in os.listdir(visualization_path):
        image_folder_path = os.path.join(visualization_path, image_folder)
        images = sorted([os.path.join(image_folder_path, img) for img in os.listdir(image_folder_path) if img.endswith(".jpg") or img.endswith(".png")])
    
        # Create video clip
        clip = ImageSequenceClip(images, fps=frame_rate)
        
        # Write the video file
        clip.write_videofile(os.path.join(output_path, image_folder + ".mp4"), codec='libx264', fps=frame_rate)

if __name__ == "__main__":
    # Define the folder containing images and the output video file name
    visualization_path = "/home/xucao2/SocialGesture/tools/visualization"
    # image_folder = '/home/xucao2/SocialGesture/tools/visualization/BEST_OF_ONE_NIGHT_ULTIMATE_WEREWOLF__Tanner_Wins_Episode_1_Game1'
    output_path = './output_vis_video'

    # Create video from images
    create_video_from_images(visualization_path, output_path)