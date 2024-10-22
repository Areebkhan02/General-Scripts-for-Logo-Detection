from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO("/home/areebadnan/Areeb_code/work/Atheritia/All_models/Large/coco_base_finetune_17logo_L.pt")

# Run inference on 'bus.jpg' with arguments
#model.predict(source="/home/areebadnan/Areeb_code/work/Atheritia/Datasets/47_logos_dataset/10_classes_final/final/split/test/images",project = '/home/areebadnan/Areeb_code/work/Atheritia/Datasets/47_logos_dataset/10_classes_final/final/split/test/images_predict',name = 'predict2', save=True, imgsz=1240, conf=0.1)

model.predict(source='/home/areebadnan/Areeb_code/work/Visua_Data/videos/17logoslinkedin.mp4', project = '/home/areebadnan/Areeb_code/work/Visua_Data/videos/',imgsz=640, name = "17logoslinkedin_predict_yolov8" , conf = 0.01, save=True)