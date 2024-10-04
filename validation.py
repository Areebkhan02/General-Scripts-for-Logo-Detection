from ultralytics import YOLO

# Load a model
model = YOLO("/home/areebadnan/Areeb_code/work/Atheritia/All_models/Large/coco_base_finetune_17logo_L.pt")

# Validate with a custom dataset
metrics = model.val(data="/home/areebadnan/Areeb_code/work/Atheritia/Datasets/17_logo_dataset_for_head_training/data.yaml", project = '/home/areebadnan/Areeb_code/work/Atheritia/Datasets/17_logo_dataset_for_head_training/val_iterations/', name = 'val_full_finetune_coco_base_17logo', imgsz=640, conf=0.1)


