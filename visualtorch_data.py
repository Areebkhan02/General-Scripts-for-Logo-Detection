from ultralytics import YOLO
import torch
import copy
import visualtorch

input_shape = (1, 3, 640, 640)


model = YOLO("yolov8n.yaml")
img = visualtorch.layered_view(model, input_shape=input_shape)


#not working with YOLO 