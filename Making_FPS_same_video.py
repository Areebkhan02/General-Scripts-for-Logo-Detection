import cv2

def match_fps(video_path_1, video_path_2, output_path_1, output_path_2):
    # Open the first video
    cap1 = cv2.VideoCapture(video_path_1)
    cap2 = cv2.VideoCapture(video_path_2)

    # Get the FPS and frame size of the first video
    fps1 = cap1.get(cv2.CAP_PROP_FPS)
    fps2 = cap2.get(cv2.CAP_PROP_FPS)

    print(f"FPS of video 1: {fps1}")
    print(f"FPS of video 2: {fps2}")

    # Get frame dimensions for both videos
    width1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
    height2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter objects to save the new videos
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for saving MP4 files
    out1 = cv2.VideoWriter(output_path_1, fourcc, fps1, (width1, height1))
    out2 = cv2.VideoWriter(output_path_2, fourcc, fps1, (width2, height2))

    # Function to convert frame rate
    def adjust_fps(cap, out, original_fps, target_fps):
        frame_interval = int(original_fps / target_fps) if original_fps > target_fps else 1
        count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # Write every 'frame_interval' frames
            if count % frame_interval == 0:
                out.write(frame)
            count += 1

    # Adjust FPS of video 1 (if needed)
    adjust_fps(cap1, out1, fps1, fps1)

    # Adjust FPS of video 2 to match video 1
    adjust_fps(cap2, out2, fps2, fps1)

    # Release everything
    cap1.release()
    cap2.release()
    out1.release()
    out2.release()
    cv2.destroyAllWindows()

# Paths for the input videos and output videos
video_path_1 = '/home/areebadnan/Areeb_code/work/Visua_Data/firefly_videos/cam2.mp4'
video_path_2 = '/home/areebadnan/Areeb_code/work/Visua_Data/firefly_videos/cam1.mp4'
output_path_1 = '/home/areebadnan/Areeb_code/work/Visua_Data/firefly_videos/output_cam1.mp4'
output_path_2 = '/home/areebadnan/Areeb_code/work/Visua_Data/firefly_videos/output_cam2.mp4'

# Call the function to match the FPS
match_fps(video_path_1, video_path_2, output_path_1, output_path_2)
