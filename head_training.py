'''

This is the script for training the head based on new architecture modification. 
Pointers 
1) This script should not be run in standard yolo environment but yolo repo should be installed and the patch added in
2) The reason for doing this is because (callbacks are added & concat head function) are in the patch 

Things to change:
1) YOLO Model path 
2) dataset path 

'''

from ultralytics import YOLO
import torch
import copy


# Initialize pretrained model
#model = YOLO('yolov8n.pt')

model = YOLO("/home/areebadnankhan/code/work/Atheritia/All_models/yolov8n_3logos_22freezed_withBN.pt")

# Keep a copy of old state dict for sanity check
old_dict = copy.deepcopy(model.state_dict())

# Add a callback to put the frozen layers in eval mode to prevent BN values
# from changing.
def put_in_eval_mode(trainer, n_layers=22):
  for i, (name, module) in enumerate(trainer.model.named_modules()):
    if name.endswith("bn") and int(name.split('.')[1]) < n_layers:
      module.eval()
      module.track_running_stats = False
      # print(name, " put in eval mode.")

model.add_callback("on_train_epoch_start", put_in_eval_mode)
model.add_callback("on_pretrain_routine_start", put_in_eval_mode)


# Train the model. Freeze the first 22 layers [0-21].
results = model.train(data='/content/datasets/data.yaml', freeze=22, epochs=100, imgsz=640)