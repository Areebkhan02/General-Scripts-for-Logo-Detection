from ultralytics import YOLO


logo_det_model = YOLO('/home/areebadnan/Areeb_code/work/Atheritia/All_models/Large/coco_base_finetune_17logo_L.pt')

results = logo_det_model.track(source="/home/areebadnan/Areeb_code/work/Atheritia/Videos/Input/2870021810804509175.mp4", project='/home/areebadnan/Desktop/data', conf=0.30, verbose=True,imgsz=1280, save=True) # persist=True,

