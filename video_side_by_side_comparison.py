# from moviepy.editor import VideoFileClip, clips_array, CompositeVideoClip, ImageClip
# from PIL import Image, ImageDraw, ImageFont
# import numpy as np
# import cv2


# def create_side_by_side_video(video1_path, video2_path, output_path):
#     # Load the video clips
#     video1 = VideoFileClip(video1_path)
#     video2 = VideoFileClip(video2_path)

#     # Ensure both videos have the same duration
#     min_duration = min(video1.duration, video2.duration)
#     video1 = video1.subclip(0, min_duration)
#     video2 = video2.subclip(0, min_duration)

#     # Create a text image for the heading
#     # heading_image = create_text_image("YOLOv8 vs YOLO11", video1.w + video2.w, 100)
#     # heading_clip = ImageClip(heading_image).set_duration(min_duration)

#     # Create a side-by-side video
#     side_by_side = clips_array([[video1, video2]])
 
#     # Combine the heading and the side-by-side video
#     final_video = CompositeVideoClip([side_by_side.set_position(("center", "bottom"))])

#     # Write the result to a file
#     final_video.write_videofile(output_path, fps=video1.fps)

# # Example usage

# path1 = "/home/areebadnan/Areeb_code/work/Visua_Data/videos/17logoslinkedin_predict_yolov8/17logoslinkedin.avi"
# path2 = "/home/areebadnan/Areeb_code/work/Visua_Data/videos/17logoslinkedin_predict/17logoslinkedin.avi"

# create_side_by_side_video(path1, path2, "/home/areebadnan/Areeb_code/work/Visua_Data/videos/comparison_video.mp4")



import cv2
from moviepy.editor import VideoFileClip, clips_array, CompositeVideoClip
import numpy as np

def add_text_to_frame(frame, text, position, font_scale=1, color=(255, 255, 255), thickness=2):
    # Make a writable copy of the frame
    frame_copy = np.array(frame, copy=True)
    
    # Use OpenCV to put text on the frame
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = position[0] - text_size[0] // 2
    text_y = position[1] + text_size[1] // 2
    cv2.putText(frame_copy, text, (text_x, text_y), font, font_scale, color, thickness, lineType=cv2.LINE_AA)
    return frame_copy

def create_side_by_side_video(video1_path, video2_path, output_path):
    # Load the video clips
    video1 = VideoFileClip(video1_path)
    video2 = VideoFileClip(video2_path)

    # Ensure both videos have the same duration
    min_duration = min(video1.duration, video2.duration)
    video1 = video1.subclip(0, min_duration)
    video2 = video2.subclip(0, min_duration)

    # Function to add text to each frame
    def add_text_to_video1(get_frame, t):
        frame = get_frame(t)
        return add_text_to_frame(frame, "YOLOv8", (frame.shape[1] // 2, 50), font_scale=2, color=(0, 0, 255), thickness=3)

    def add_text_to_video2(get_frame, t):
        frame = get_frame(t)
        return add_text_to_frame(frame, "YOLO11", (frame.shape[1] // 2, 50), font_scale=2, color=(0, 0, 255), thickness=3)

    # Apply the text overlay to each video
    video1_with_text = video1.fl(add_text_to_video1)
    video2_with_text = video2.fl(add_text_to_video2)

    # Create a side-by-side video
    side_by_side = clips_array([[video1_with_text, video2_with_text]])

    # Write the result to a file
    side_by_side.write_videofile(output_path, fps=video1.fps)

# Example usage
path1 = "/home/areebadnan/Areeb_code/work/Visua_Data/videos/17logoslinkedin_predict_yolov8/17logoslinkedin.avi"
path2 = "/home/areebadnan/Areeb_code/work/Visua_Data/videos/17logoslinkedin_predict/17logoslinkedin.avi"

create_side_by_side_video(path1, path2, "/home/areebadnan/Areeb_code/work/Visua_Data/videos/comparison_video2.mp4")
