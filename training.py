import ultralytics
from ultralytics import YOLO

#model = YOLO("yolo11l.pt")

#result=model.train(data = "/home/areebadnan/Areeb_code/work/Atheritia/Datasets/17_logo_dataset_for_head_training/data.yaml", epochs = 200, imgsz = 640, batch = -1, project = "/home/areebadnan/Areeb_code/work/Atheritia/Datasets/17_logo_dataset_for_head_training/train_iterations/", name = "yolov11l_17logo_train_fullfinetune_cocobase_200epochs")

model = YOLO("/home/areebadnan/Areeb_code/work/Atheritia/Datasets/17_logo_dataset_for_head_training/train_iterations/yolov11l_17logo_train_fullfinetune_cocobase_200epochs/weights/last.pt")

result = model.train(resume = True)