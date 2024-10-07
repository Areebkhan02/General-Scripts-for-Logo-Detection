from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO("/home/areebadnan/Areeb_code/work/Atheritia/All_models/Large/14layers_freeze_3heads_merged_64_cls_updated(1).pt")

# Run inference on 'bus.jpg' with arguments
#model.predict(source="/home/areebadnan/Areeb_code/work/Atheritia/Datasets/47_logos_dataset/10_classes_final/final/split/test/images",project = '/home/areebadnan/Areeb_code/work/Atheritia/Datasets/47_logos_dataset/10_classes_final/final/split/test/images_predict',name = 'predict2', save=True, imgsz=1240, conf=0.1)

model.predict(source='/home/areebadnan/Areeb_code/work/Atheritia/Datasets/47_logos_dataset/10_classes_final/final/split2/test/test_videos_for_pipeline/video1.mp4', project = '/home/areebadnan/Areeb_code/work/Atheritia/Datasets/47_logos_dataset/10_classes_final/final/split2/test/test_videos_output_for_pipeline/',imgsz=640, name = "pred1_640" , conf = 0.01, save=True)