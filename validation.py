from ultralytics import YOLO

# Load a model
model = YOLO("/home/areebadnan/Areeb_code/work/Atheritia/All_models/Large/47_37_retrainbest.pt")

# Validate with a custom dataset
metrics = model.val(data="/home/areebadnan/Areeb_code/work/Atheritia/Datasets/3heads_merged_dataset/47_37_logos_dataset/data.yaml", project = '/home/areebadnan/Areeb_code/work/Atheritia/Datasets/3heads_merged_dataset/47_37_logos_dataset/val_iterations/', name = 'val2_648_47_37_model', imgsz=648, conf=0.1)


