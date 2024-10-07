import os 
import supervision as sv  # Library for handling video processing and annotation
import numpy as np  
from ultralytics import YOLO  
from tqdm import tqdm

# Path to the input video file
VIDEO_PATH = "/home/jansher/Athletia/Visua Data/Scripts and Data/visua_folio3_result/videos/614a1318181b3ef94faeb847.mp4"
# Directory to store the processed output video
OUTPUT_DIR = "/home/jansher/Athletia/Visua Data/Scripts and Data/visua_folio3_result/videos_folio3_result/"

# Extract the input video's name without the extension for naming the output file
video_name = os.path.splitext(os.path.basename(VIDEO_PATH))[0]

# Construct the output path with the same name as the input video
OUTPUT_PATH = os.path.join(OUTPUT_DIR, f"{video_name}_processed.mp4")

# Create the output directory if it doesn't already exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load the YOLO model with the specified weights
model = YOLO('/home/jansher/Athletia/17_logo_detection/iteration_2/ec2_training/2nd_training/train/weights/best_8_july_24.pt')

# Get video information such as frame count, frame rate, etc.
video_info = sv.VideoInfo.from_video_path(VIDEO_PATH)

def process_frame(frame: np.ndarray, _) -> np.ndarray:
    """
    Process a single video frame: run detection and annotate detected objects.

    Args:
        frame (np.ndarray): The input video frame to process.
        _ : Unused parameter (placeholder for frame index).

    Returns:
        np.ndarray: The processed video frame with annotations.
    """
    # Perform object tracking on the current frame
    results = model.predict(frame, verbose=False)[0]
    # Convert YOLO results to detections
    detections = sv.Detections.from_ultralytics(results)
    # Create a box annotator for drawing bounding boxes
    box_annotator = sv.BoxAnnotator(thickness=2, text_thickness=1, text_scale=0.5)

    # Create labels for each detection with class name and confidence score
    # labels = [f"{model.names[class_id]} {confidence:0.2f}" for _, _, confidence, class_id, _ in detections]
    labels = [f"{model.names[class_id]}" for _, _, confidence, class_id, _ in detections]

    # Annotate the frame with bounding boxes and labels
    frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels)

    return frame

# Count the total number of frames in the video for the progress bar
num_frames = int(video_info.total_frames)

# Process the video with a tqdm progress bar
with tqdm(total=num_frames) as pbar:
    def callback_with_progress(frame, frame_index):
        """
        Callback function to process each frame and update the progress bar.

        Args:
            frame: The current video frame.
            frame_index: The index of the current frame.

        Returns:
            The processed frame.
        """
        # Process the current frame
        processed_frame = process_frame(frame, frame_index)
        # Update the progress bar by one step
        pbar.update(1)
        return processed_frame

    # Process the entire video and save the output
    sv.process_video(source_path=VIDEO_PATH, target_path=OUTPUT_PATH, callback=callback_with_progress)
