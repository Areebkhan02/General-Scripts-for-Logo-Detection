# -*- coding: utf-8 -*-

import cv2
import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torchvision 
from torchvision import transforms
from torchvision import models
from ultralytics import YOLO
from tqdm import tqdm
#from efficientnet_pytorch import EfficientNet
import difflib

# def calculate_iou(box1, box2):
#     x1, y1, w1, h1 = box1
#     x2, y2, w2, h2 = box2

#     # Calculate coordinates of intersection rectangle
#     x_intersection = max(x1, x2)
#     y_intersection = max(y1, y2)
#     w_intersection = min(x1 + w1, x2 + w2) - x_intersection
#     h_intersection = min(y1 + h1, y2 + h2) - y_intersection

#     # Calculate area of intersection rectangle
#     area_intersection = max(0, w_intersection) * max(0, h_intersection)

#     # Calculate areas of the two bounding boxes
#     area_box1 = w1 * h1
#     area_box2 = w2 * h2

#     # Calculate IoU
#     iou = area_intersection / (area_box1 + area_box2 - area_intersection)
#     return iou

def calculate_iou(box1, box2):
    # Unpack the coordinates for the intersection rectangle
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    # Calculate the area of intersection rectangle
    inter_width = max(0, x2_inter - x1_inter)
    inter_height = max(0, y2_inter - y1_inter)
    inter_area = inter_width * inter_height

    # Calculate the area of both the prediction and ground-truth rectangles
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Calculate the Union area by using Formula: Union(A,B) = A + B - Inter(A,B)
    union_area = box1_area + box2_area - inter_area

    # Compute theyo IoU
    iou = inter_area / union_area if union_area else 0
    return iou


def pre_process(image):
    input_size = (80, 80)  # Match this to your training input size
    image = cv2.resize(image, input_size)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    image = image.astype(np.float32) / 255.0  # Normalize to [0, 1]

    # Apply the same normalization as during training
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    # Convert the NumPy array to a PyTorch tensor and apply normalization
    image_tensor = torch.from_numpy(image).permute(2, 0, 1)  # Channels first (C, H, W)
    image_tensor = normalize(image_tensor)

    # Add a batch dimension if needed (depends on your model's input requirements)
    image_tensor = image_tensor.unsqueeze(0)
    return image_tensor


# Load the YOLOv8 model

logo_det_model = YOLO('/home/areebadnan/Areeb_code/work/Atheritia/All_models/Large/coco_base_finetune_17logo_L.pt')
logo_source_model = YOLO('/home/areebadnan/Areeb_code/work/Atheritia/All_models/Source_models/logo_source_model.pt')

# use OpenCV to read the video frames 

#video_path = "/home/muhammadanassiddiqui/Downloads/cropped_video.mp4" #bayer logo
#video_path = "/home/muhammadanassiddiqui/video1.mp4"
#video_path = "/home/muhammadanassiddiqui/0227 (1).mp4"
#video_path = '/home/muhammadanassiddiqui/input_video_2.mp4'
#video_path =  "27th_sep/videos/6207f5ae7918cadc89da1dfa.mp4"
video_path = "Videos/Input/input_video2.mp4"
cap = cv2.VideoCapture(video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

output_video_path = "/home/areebadnan/Areeb_code/work/Atheritia/Videos/Output/output_video2_track.mp4"
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))


# Initialize variables to store data
total_time = 0
results_dict = {}
detections = len(results_dict)


num_frames = 0
total_time = 0
results_dict = {}
id_start_frame = {}  # Dictionary to store the start frame for each unique ID
id_end_frame = {}    # Dictionary to store the end frame for each unique ID
id_frame_duration = {}  

#class_names = ["logo"]

# Initialize tqdm with the total number of frames

pbar = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), desc="Processing video")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success :
        num_frames += 1 
        t1 = time.time()
        # Update tqdm progress bar
        pbar.update(1)
        # Run YOLOv8 inference on the frame
        results = logo_det_model.track(frame, persist=True, conf=0.30, verbose=False,imgsz=1280)                # Enable score fusion)
        boxes = results[0].boxes

        if len(boxes) > 0:
                    # Convert class tensor to CPU and extract class IDs
                    class_ids = boxes.cls.cpu().numpy().astype(int)
                    print("ddd",class_ids)

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
           
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            conf = results[0].boxes.conf.tolist()

            for box, id,cl_id,conf in zip(boxes, ids,class_ids,conf):
                int_id = int(id)
                if int_id not in results_dict:
                    print(conf,"-----------------------")
                    results_dict[int_id] = {"bounding_box":[],"source":"Unknown", "conf":conf}
                    id_start_frame[int_id] = num_frames  # Set start frame for new unique ID
                    class_id = cl_id
                    
                    # Get class label using class ID
                    class_label = logo_det_model.names[class_id] if class_id < len(logo_det_model.names) else f'Class_{class_id}'
                    print(logo_det_model.names[class_id])
                    results_dict[int_id]["logo_name"] = class_label
                      # Crop the image using the bounding box coordinates
                    cropped_img = frame[box[1]:box[3], box[0]:box[2]]
                    cropped_img_tensor = pre_process(cropped_img)
                    # Load the image using OpenCV
                thickness = 2  # You can adjust the thickness of the bounding box
                color = (0, 0, 255) 
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, thickness)
                results_dict[int_id]["bounding_box"].append(box.tolist())

                # Add text (logo name) above the bounding box
                results_dict[int_id]["source"] = "Unknown" 
                text = results_dict[int_id]["logo_name"]
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1
                text_color = (255, 255, 255)  # White color for text
                text_thickness = 1
                text_size = cv2.getTextSize(text, font, font_scale, text_thickness)[0]
                text_x = box[0] + (box[2] - box[0]) // 2 - text_size[0] // 2
                text_y = box[1] - 5  # Adjust this value to position the text properly
                cv2.rectangle(frame, (text_x - 5, text_y - text_size[1] - 10),
                                      (text_x + text_size[0] + 10, text_y + 5), color, -1)
                src_text = results_dict[int_id]["source"]
                src_text_size = cv2.getTextSize(src_text, font, font_scale, text_thickness)[0]
                cv2.putText(frame, text, (text_x, text_y), font, font_scale, text_color, text_thickness)
                if src_text != 'Unknown':
                     cv2.rectangle(frame, (text_x+text_size[0]+20, text_y - text_size[1] - 5),
                                      (text_x + text_size[0]+src_text_size[0] + 10, text_y + 5), color, -1)
                     cv2.putText(frame, src_text, (text_x+text_size[0]+20, text_y), font, font_scale, text_color, text_thickness)

            
                id_end_frame[int_id] = num_frames  # Update end frame for the unique ID

                    # review these lines to add sorce in on file 

            if len(results_dict) > detections: #new detection 
                detections = len(results_dict)
                src_res = logo_source_model(frame, verbose=False, conf=0.2,save=True)
                src_boxes = src_res[0].boxes.xyxy.cpu().numpy().astype(int)   
                src_boxe_bb = src_res[0].boxes.xywh.cpu().numpy().astype(int)     
                src_class_ids = src_res[0].boxes.cls.cpu().numpy().astype(int)
                all_src_names = src_res[0].names
                det_src_names = [list(all_src_names.items())[class_id][1] for class_id in src_class_ids]
                    
                for logo_box, id in zip(boxes, ids):
                    int_id = int(id)
                    max_iou = 0
                    logo_src = None
                    for src_box, src_name in zip(src_boxes, det_src_names):
                        iou = calculate_iou(src_box, logo_box)
                        #print("iou: ", iou)
                        if max_iou < iou:
                            max_iou = iou
                            logo_src = src_name
                    #print(max_iou, int_id)

                    if max_iou > 0.05:
                        results_dict[int_id]["source"] = logo_src 
                       # Add text (logo source) below the bounding box
        #    Add text (logo source) below the bounding box
                    #   print(logo_src)
                       

                       # results_dict[int_id]["logo_name"] = 'logo'

                    
        # Write the frame with bounding boxes to the output video
        out.write(frame)


        t2 = time.time()
        t = t2 - t1
        total_time += t

    else:
        break


# Close tqdm progress bar
pbar.close()
# Release the video capture and writer objects
cap.release()
out.release()

# Calculate frame duration for each unique ID and add it to the results_dict
for id, start_frame in id_start_frame.items():
    end_frame = id_end_frame[id]
    frame_duration = end_frame - start_frame + 1
    results_dict[id]["start_frame"] = start_frame
    results_dict[id]["end_frame"] = end_frame
    results_dict[id]["frame_duration"] = frame_duration

json_data_list = []

for tracking_id, item in results_dict.items():
    #print(tracking_id, item)
    json_item = {
        'tracking_id': tracking_id,
        'logo_name': item['logo_name'],
        'start_time': item['start_frame'] / fps,
        'end_time': item['end_frame'] / fps,
        'time_duration(s)': item['frame_duration'] / fps,
        'logo_source': item['source'],
        'confidence': item['conf'],
        'bounding_box': item['bounding_box']
    }
    json_data_list.append(json_item)

# Write the JSON data to the file
with open('/home/areebadnan/Areeb_code/work/Atheritia/Videos/Output/output_video2_track.json', 'w') as json_file:
    json.dump(json_data_list, json_file, indent=4)

print(f"Number of frames: {num_frames} sec")
print(f"Total processing time: {total_time}")
print(f"Average processing time per frame: {total_time / num_frames}")
print(f"Number of unique IDs detected: {len(results_dict)}")
