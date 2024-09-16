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

video_path = "/home/areebadnan/Areeb_code/work/Atheritia/Videos/Input/2870021810804509175.mp4"
cap = cv2.VideoCapture(video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

output_video_path = "/home/areebadnan/Areeb_code/work/Atheritia/Videos/Output/output1_predict.mp4"
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

total_time = 0
results_dict = {}
detections = len(results_dict)
num_frames = 0

pbar = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), desc="Processing video")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

while cap.isOpened():
    success, frame = cap.read()

    if success:
        num_frames += 1
        t1 = time.time()
        pbar.update(1)

        results = logo_det_model.predict(frame, conf=0.30, verbose=False, imgsz=1280)
        boxes = results[0].boxes

        if len(boxes) > 0:
            class_ids = boxes.cls.cpu().numpy().astype(int)
            print("Detected class_ids:", class_ids)

            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            confs = results[0].boxes.conf.tolist()

            for box, class_id, conf in zip(boxes, class_ids, confs):
                if class_id not in results_dict:
                    print(f"Confidence: {conf}")
                    results_dict[class_id] = {"bounding_box": [], "source": "Unknown", "conf": conf, "logo_name": ""}
                    class_label = logo_det_model.names[class_id] if class_id < len(logo_det_model.names) else f'Class_{class_id}'
                    print("Detected class label:", class_label)
                    results_dict[class_id]["logo_name"] = class_label
                    
                results_dict[class_id]["bounding_box"].append(box.tolist())

                cropped_img = frame[box[1]:box[3], box[0]:box[2]]
                cropped_img_tensor = pre_process(cropped_img)

                thickness = 2
                color = (0, 0, 255)
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, thickness)

                text = results_dict[class_id]["logo_name"]
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1
                text_color = (255, 255, 255)
                text_thickness = 1
                text_size = cv2.getTextSize(text, font, font_scale, text_thickness)[0]
                text_x = box[0] + (box[2] - box[0]) // 2 - text_size[0] // 2
                text_y = box[1] - 5
                cv2.rectangle(frame, (text_x - 5, text_y - text_size[1] - 10),
                              (text_x + text_size[0] + 10, text_y + 5), color, -1)
                cv2.putText(frame, text, (text_x, text_y), font, font_scale, text_color, text_thickness)

            if len(results_dict) > detections:
                detections = len(results_dict)
                src_res = logo_source_model(frame, verbose=False, conf=0.2)
                src_boxes = src_res[0].boxes.xyxy.cpu().numpy().astype(int)
                src_class_ids = src_res[0].boxes.cls.cpu().numpy().astype(int)
                all_src_names = src_res[0].names
                det_src_names = [list(all_src_names.items())[class_id][1] for class_id in src_class_ids]

                for box, class_id in zip(boxes, class_ids):
                    max_iou = 0
                    logo_src = None
                    for src_box, src_name in zip(src_boxes, det_src_names):
                        iou = calculate_iou(src_box, box)
                        if max_iou < iou:
                            max_iou = iou
                            logo_src = src_name
                    if max_iou > 0.05:
                        results_dict[class_id]["source"] = logo_src

                        src_text = results_dict[class_id]["source"]
                        src_text_size = cv2.getTextSize(src_text, font, font_scale, text_thickness)[0]
                        cv2.putText(frame, src_text, (text_x + text_size[0] + 20, text_y), font, font_scale, text_color, text_thickness)

        out.write(frame)

        t2 = time.time()
        t = t2 - t1
        total_time += t

    else:
        break

pbar.close()
cap.release()
out.release()

# Convert int64 to int for JSON serialization
json_data_list = []

for class_id, item in results_dict.items():
    json_item = {
        'class_id': int(class_id),
        'logo_name': item['logo_name'],
        'confidence': item['conf'],
        'logo_source': item['source'],
        'bounding_box': item['bounding_box']
    }
    json_data_list.append(json_item)

# Write the JSON data to a file
with open('/home/areebadnan/Areeb_code/work/Atheritia/Videos/Output/output_predict.json', 'w') as json_file:
    json.dump(json_data_list, json_file, indent=4)

print(f"Number of frames: {num_frames}")
print(f"Total processing time: {total_time}")
print(f"Average processing time per frame: {total_time / num_frames}")
print(f"Number of unique classes detected: {len(results_dict)}")
