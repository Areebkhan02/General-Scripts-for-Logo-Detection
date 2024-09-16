import os
import shutil
from tqdm import tqdm  # Import tqdm for the progress bar

# Define the paths to your images and labels folders
images_folder = "/home/areebadnan/Areeb_code/work/Visua_Data/ground_truth/images"  # e.g., "images"
labels_folder = "/home/areebadnan/Areeb_code/work/Visua_Data/ground_truth/labels"  # e.g., "labels"
output_base_folder = "/home/areebadnan/Areeb_code/work/Visua_Data/output_videos"  # Output folder to store the organized folders

# Ensure the output folder exists
os.makedirs(output_base_folder, exist_ok=True)

def organize_files_by_video(base_folder, target_folder):
    # List all files in the given folder
    files = [filename for filename in os.listdir(base_folder) if "_" in filename]

    # Use tqdm to create a progress bar
    for filename in tqdm(files, desc=f"Processing {base_folder}"):
        unique_id = filename.split("_")[0]  # Extract the unique ID before the underscore
        video_folder = os.path.join(target_folder, f"{unique_id}")

        # Create the video-specific folder and its subfolders
        os.makedirs(os.path.join(video_folder, "images"), exist_ok=True)
        os.makedirs(os.path.join(video_folder, "labels"), exist_ok=True)

        # Determine the type of file (image or label)
        if base_folder == images_folder:
            destination_folder = os.path.join(video_folder, "images")
        else:
            destination_folder = os.path.join(video_folder, "labels")

        # Copy the file to the appropriate folder
        shutil.copy(os.path.join(base_folder, filename), os.path.join(destination_folder, filename))

# Organize images and labels
organize_files_by_video(images_folder, output_base_folder)
organize_files_by_video(labels_folder, output_base_folder)

print("Files have been organized successfully!")
