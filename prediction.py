from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO("/home/areebadnan/Areeb_code/work/Atheritia/All_models/Large/47_37_retrainbest.pt")

# Run inference on 'bus.jpg' with arguments
#model.predict(source="/home/areebadnan/Areeb_code/work/Atheritia/Datasets/47_logos_dataset/10_classes_final/final/split/test/images",project = '/home/areebadnan/Areeb_code/work/Atheritia/Datasets/47_logos_dataset/10_classes_final/final/split/test/images_predict',name = 'predict2', save=True, imgsz=1240, conf=0.1)

model.predict(source='Datasets/3heads_merged_dataset/47_37_val_logos_dataset/issue_checking/ERGO/images', project = 'Datasets/3heads_merged_dataset/47_37_val_logos_dataset/issue_checking/ERGO/prediction',imgsz=640, name = "pred1_640" , conf = 0.01, save=True)