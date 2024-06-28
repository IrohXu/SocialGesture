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


def yolov10_inference(video, video_id, out_path, image_size, conf_threshold):
    # model = ort.InferenceSession('/home/xucao2/checkpoints/yolov10b.onnx')
    model = YOLOv10.from_pretrained('jameslahm/yolov10x')

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

        # image = cv2.resize(frame, (640, 640))
        # input_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # # Assuming the model expects a certain input size
        # input_tensor = input_image.astype(np.float32) / 255.0
        # input_tensor = input_tensor.transpose(2, 0, 1)  # Change to channel-first format
        # input_tensor = np.expand_dims(input_tensor, axis=0)

        # # Run the model
        # inputs = {model.get_inputs()[0].name: input_tensor}
        # outputs = model.run(None, inputs)
        #
        # # Process the output
        # output = outputs[0]  # Assuming the output is [1, 300, 6]
        # output = output[output[:,:,4]>conf_threshold]
        # output = output[output[:,5]==0]
        #
        # # Extract bounding boxes and probabilities
        # boxes = output[:, :4]
        # probabilities = output[:, 4]
        
        frame_info = {
            "frame": frame_id,
            "segments_info": []
        }

        # Find the box with the highest probability
        # max_prob_index = np.argmax(probabilities)
        # x1, y1, x2, y2 = boxes[max_prob_index]
        
        for box in boxes_person:
            x1, y1, x2, y2 = box
            frame_info["segments_info"].append({
                "category_id": 0,
                "bbox": [
                    int(x1), int(y1), int(x2), int(y2)
                ]
            })
        
        # x1, y1, x2, y2 = boxes[0]

        # Draw rectangle on the image
        # start_point = (int(x1), int(y1 * 360 /640))
        # end_point = (int(x2), int(y2 * 360 /640))
        # output_image = cv2.rectangle(image.copy(), start_point, end_point, 1)
        
        output_json["annotations"].append(frame_info)
        
        frame_id += 1
        
        # plt.imsave('output.jpg', cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))

        # results = model.predict(source=frame, imgsz=image_size, conf=conf_threshold)
        # annotated_frame = results[0].plot()
        # out.write(annotated_frame)
        # with open(os.path.join(out_path, video_id + '.json'), 'w') as f:
        #     json.dump(output_json, f, indent=4)  # prettier json

    cap.release()
    with open(os.path.join(out_path, video_id + '.json'), 'w') as f:
        json.dump(output_json, f, indent=4)  # prettier json

    return None

image_size = 640
conf_threshold = 0.6

for video_name in os.listdir("/home/xucao2/SocialGesture/data_sources/pilot_dataset/videos"):
    video = os.path.join("/home/xucao2/SocialGesture/data_sources/pilot_dataset/videos", video_name)
    video_id = video_name.split(".")[0]

    yolov10_inference(video, video_id, "/home/xucao2/SocialGesture/tools/yolov10/annotation_output", image_size, conf_threshold)

# yolo track model=jameslahm/yolov10b source=/home/xucao2/SocialGesture/data_sources/pilot_dataset/videos/Anger_Lies_Swap_City_Whats_not_to_love__ONE_NIGHT_ULTIMATE_WEREWOLF_Game1.mp4 half conf=0.50