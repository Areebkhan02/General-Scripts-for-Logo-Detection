import cv2
import json
from tqdm import tqdm

# Load JSON file
with open('Atheritia/Videos/Output/output_3_predict.json') as f:
    annotations = json.load(f)

# Open the video file
video = cv2.VideoCapture('Atheritia/Videos/Input/614a1318181b3ef94faeb847.mp4')

# Get video properties
fps = video.get(cv2.CAP_PROP_FPS)
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('/home/areebadnan/Areeb_code/work/Atheritia/Videos/Output/output_json_1.mp4', fourcc, fps, (width, height))

# Progress bar setup
pbar = tqdm(total=total_frames, desc="Processing frames", unit="frame")

# Loop over each frame
frame_id = 0
while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break

    # Process annotations
    for annotation in annotations:
        start_frame = int(annotation['start_time'] * fps)
        end_frame = int(annotation['end_time'] * fps)

        if start_frame <= frame_id <= end_frame:
            bbox = annotation['bounding_box'][frame_id - start_frame]
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            label = f"{annotation['logo_name']} {annotation['confidence']:.2f}"
            cv2.putText(frame, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Write the frame to the output video
    out.write(frame)

    # Update the progress bar
    pbar.update(1)

    frame_id += 1

# Release resources
video.release()
out.release()
pbar.close()

# Attempt to destroy all OpenCV windows (if any were opened)
try:
    cv2.destroyAllWindows()
except cv2.error as e:
    print(f"Warning: Unable to destroy windows. Error: {e}")
