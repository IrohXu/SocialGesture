import os
from yolov10.ultralytics import YOLOv10
import tempfile
import onnxruntime as ort
import numpy as np
import cv2
import matplotlib.pyplot as plt
import json

# Load the ONNX model
# session = ort.InferenceSession('/home/xucao2/checkpoints/yolov10x.onnx')


def yolov10_inference(video, video_id, out_path, visualization_path, image_size, conf_threshold):
    # model = ort.InferenceSession('/home/xucao2/checkpoints/yolov10b.onnx')
    model = YOLOv10.from_pretrained('jameslahm/yolov10x')
    
    if os.path.exists(os.path.join(visualization_path, video_id)) == False:
        os.makedirs(os.path.join(visualization_path, video_id))

    output_json = {
        "video_id": video_id,
        "width": 640,
        "height": 360,
        "annotations": []
    }

    video_path = tempfile.mktemp(suffix=".webm")
    with open(video_path, "wb") as f:
        with open(video, "rb") as g:
            f.write(g.read())

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_id = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        input_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        outputs = model.predict(input_image, verbose=False)
        output = outputs[0]
        boxes = output.boxes
        boxes_person = boxes.xyxy[boxes.cls==0]
        
        frame_info = {
            "frame": frame_id,
            "segments_info": []
        }

        # Find the box with the highest probability
        # max_prob_index = np.argmax(probabilities)
        # x1, y1, x2, y2 = boxes[max_prob_index]
        
        output_image = input_image.copy()
        
        for box in boxes_person:
            x1, y1, x2, y2 = box
            frame_info["segments_info"].append({
                "category_id": 0,
                "bbox": [
                    int(x1), int(y1), int(x2), int(y2)
                ]
            })

        # Draw rectangle on the image
            start_point = (int(x1), int(y1))
            end_point = (int(x2), int(y2))
            output_image = cv2.rectangle(output_image, start_point, end_point, 1)
        
        output_json["annotations"].append(frame_info)
        
        frame_id += 1
        
        plt.imsave(os.path.join(visualization_path, video_id, str(frame_id) + '.png'), cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))

    cap.release()
    with open(os.path.join(out_path, video_id + '.json'), 'w') as f:
        json.dump(output_json, f, indent=4)  # prettier json

    return None

image_size = 640
conf_threshold = 0.9

for video_name in os.listdir("/home/xucao2/SocialGesture/data_sources/pilot_dataset/videos"):
    video = os.path.join("/home/xucao2/SocialGesture/data_sources/pilot_dataset/videos", video_name)
    video_id = video_name.split(".")[0]

    yolov10_inference(video, video_id, "/home/xucao2/SocialGesture/tools/new_output", "/home/xucao2/SocialGesture/tools/visualization", image_size, conf_threshold)

# yolo track model=jameslahm/yolov10b source=/home/xucao2/SocialGesture/data_sources/pilot_dataset/videos/Anger_Lies_Swap_City_Whats_not_to_love__ONE_NIGHT_ULTIMATE_WEREWOLF_Game1.mp4 half conf=0.50