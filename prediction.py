from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO("/home/areebadnan/Areeb_code/work/Atheritia/47_(10)_logos_head_17logosbase_L_3/train/weights/best.pt")

# Run inference on 'bus.jpg' with arguments
#model.predict(source="/home/areebadnan/Areeb_code/work/Atheritia/Datasets/47_logos_dataset/10_classes_final/final/split/test/images",project = '/home/areebadnan/Areeb_code/work/Atheritia/Datasets/47_logos_dataset/10_classes_final/final/split/test/images_predict',name = 'predict2', save=True, imgsz=1240, conf=0.1)

model.predict(source='/home/areebadnan/Areeb_code/work/Atheritia/Datasets/47_logos_dataset/10_classes_final/final/split2/test/BetWay_Issue/images', project = '/home/areebadnan/Areeb_code/work/Atheritia/Datasets/47_logos_dataset/10_classes_final/final/split2/test/BetWay_Issue/prediction',imgsz=1248, name = "pred2" , conf = 0.01, save=True)