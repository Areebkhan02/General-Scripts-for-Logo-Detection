import cv2
import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from ultralytics import YOLO
from tqdm import tqdm


def calculate_iou(box1, box2):
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    inter_width = max(0, x2_inter - x1_inter)
    inter_height = max(0, y2_inter - y1_inter)
    inter_area = inter_width * inter_height

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area

    iou = inter_area / union_area if union_area else 0
    return iou

def pre_process(image):
    input_size = (80, 80)
    image = cv2.resize(image, input_size)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    image_tensor = torch.from_numpy(image).permute(2, 0, 1)
    image_tensor = normalize(image_tensor)
    image_tensor = image_tensor.unsqueeze(0)
    return image_tensor

logo_det_model = YOLO('/home/areebadnan/Areeb_code/work/Atheritia/All_models/Large/coco_base_finetune_17logo_L.pt')
logo_source_model = YOLO('/home/areebadnan/Areeb_code/work/Atheritia/All_models/Source_models/logo_source_model.pt')

video_path = "Videos/Input/input_video3.mp4"
cap = cv2.VideoCapture(video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

output_video_path = "/home/areebadnan/Areeb_code/work/Atheritia/Videos/Output/output_video3_predict.mp4"
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

results_dict = {}
frame_detections = {}
object_counter = 0

pbar = tqdm(total=total_frames, desc="Processing video")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

while cap.isOpened():
    success, frame = cap.read()

    if success:
        current_frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        current_time = current_frame_num / fps

        results = logo_det_model.predict(frame, conf=0.30, verbose=False, imgsz=1280)
        boxes = results[0].boxes

        if len(boxes) > 0:
            class_ids = boxes.cls.cpu().numpy().astype(int)
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            confs = results[0].boxes.conf.tolist()

            frame_detections[current_frame_num] = []

            for box, class_id, conf in zip(boxes, class_ids, confs):
                # Drawing the bounding box
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                # Constructing the text for class name and confidence score
                text = f"{logo_det_model.names[class_id]}: {conf:.2f}"
                # Putting the text above the bounding box
                cv2.putText(frame, text, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                logo_data = {
                    "class_id": class_id,
                    "bbox": box.tolist(),
                    "conf": conf,
                    "start_time": current_time,
                    "end_time": current_time,  # Initially set end_time to start_time
                    "object_id": None
                }
                frame_detections[current_frame_num].append(logo_data)

        out.write(frame)
        pbar.update(1)
    else:
        break

pbar.close()
cap.release()
out.release()

# Now process the frame_detections dictionary to calculate time durations and assign object IDs
json_output = []
for frame_num, detections in frame_detections.items():
    current_time = frame_num / fps
    for detection in detections:
        class_id = detection["class_id"]
        logo_name = logo_det_model.names[class_id]

        # Check if this logo has been detected before
        found_match = False
        for obj in json_output:
            if obj["logo_name"] == logo_name:
                if current_time - obj["end_time"] <= 5:  # Within 5 seconds
                    obj["end_time"] = current_time
                    obj["time_duration(s)"] = obj["end_time"] - obj["start_time"]
                    obj["bounding_box"].append(detection["bbox"])
                    found_match = True
                    break

        if not found_match:
            object_counter += 1
            detection["object_id"] = object_counter
            detection["logo_name"] = logo_name
            detection["time_duration(s)"] = 0
            detection["bounding_box"] = [detection["bbox"]]
            json_output.append(detection)

# Convert int64 to int for JSON serialization
json_data_list = []
for item in json_output:
    json_item = {
        "object_id": item["object_id"],
        "logo_name": logo_det_model.names[item["class_id"]],
        "start_time": item["start_time"],
        "end_time": item["end_time"],
        "time_duration(s)": item["end_time"] - item["start_time"],
        "logo_source": item.get("source", "Unknown"),
        "confidence": item["conf"],
        "bounding_box": item["bounding_box"]
    }
    json_data_list.append(json_item)

# Write the JSON data to a file
with open('/home/areebadnan/Areeb_code/work/Atheritia/Videos/Output/output_video3_predict.json', 'w') as json_file:
    json.dump(json_data_list, json_file, indent=4)

print("Processing completed!")
