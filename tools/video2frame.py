import cv2
import os

def video_to_frames(video_path, output_folder):
    # Check if the video file exists
    if not os.path.exists(video_path):
        print(f"Video {video_path} not found!")
        return

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Load the video
    cap = cv2.VideoCapture(video_path)

    # Get the total frame count
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames: {total_frames}")

    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Construct the output filename for the frame
        frame_filename = os.path.join(output_folder, f"frame_{frame_num:04d}.png")

        # Save the frame as an image
        cv2.imwrite(frame_filename, frame)
        print(f"Saved {frame_filename}")

        frame_num += 1

    # Release the video capture object
    cap.release()
    print("Finished extracting frames!")

# Example usage:

werewolf_dir = "/home/xucao2/tool/youtube/output"
save_dir = "/home/xucao2/tool/Tracking-Anything-with-DEVA/video_frames"

exist_file_list = os.listdir(save_dir)

for file in os.listdir(werewolf_dir):
    video_path = os.path.join(werewolf_dir, file)
    video_name = file.split(".")[0].split(" ")
    video_folder = "_".join(video_name)
    if video_folder in exist_file_list:
        continue
    video_to_frames(video_path, os.path.join(save_dir, video_folder))