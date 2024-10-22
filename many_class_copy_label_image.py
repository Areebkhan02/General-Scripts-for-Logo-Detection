import os
import shutil
from tqdm import tqdm

def copy_files(images_folder, labels_folder, class_names, dest_images_folder, dest_labels_folder):
    """
    Copies text files and their corresponding images to new folders based on given class names.

    Args:
        images_folder (str): Path to the folder containing images.
        labels_folder (str): Path to the folder containing YOLO label files.
        class_names (list): List of class names to search for in the YOLO label files.
        dest_images_folder (str): Path to the destination folder for images.
        dest_labels_folder (str): Path to the destination folder for label files.

    Returns:
        None
    """

    # Create destination directories if they don't exist
    os.makedirs(dest_images_folder, exist_ok=True)
    os.makedirs(dest_labels_folder, exist_ok=True)

    label_files = [f for f in os.listdir(labels_folder) if f.endswith(".txt")]

    # Initialize progress bar
    with tqdm(total=len(label_files), desc="Processing files") as pbar:
        # Iterate over all txt files in the labels folder
        for label_file in label_files:
            label_file_path = os.path.join(labels_folder, label_file)

            # Read the content of the label file
            with open(label_file_path, 'r') as file:
                lines = file.readlines()

            # Check if any of the class names exist in the label file
            class_found = False
            for line in lines:
                line_parts = line.strip().split()
                if len(line_parts) > 0 and line_parts[0] in class_names:
                    class_found = True
                    break

            # If any class is found, copy the label file and corresponding image
            if class_found:
                # Copy the label file
                dest_label_path = os.path.join(dest_labels_folder, label_file)
                shutil.copy(label_file_path, dest_label_path)
                print(f"Copied label file: {label_file}")

                # Get the corresponding image name (same as label file but with .jpg extension)
                image_file = label_file.replace(".txt", ".jpg")
                image_file_path = os.path.join(images_folder, image_file)

                # Copy the corresponding image file if it exists
                if os.path.exists(image_file_path):
                    dest_image_path = os.path.join(dest_images_folder, image_file)
                    shutil.copy(image_file_path, dest_image_path)
                    print(f"Copied image file: {image_file}")
                else:
                    print(f"Image file not found for {label_file}")

            # Update the progress bar
            pbar.update(1)

if __name__ == "__main__":
    # Input from the user
    images_folder = '/home/areebadnan/Areeb_code/work/Atheritia/Datasets/47_logos_dataset/10_classes_final/final/split2/47_(10)_cleaned_images/images'
    labels_folder = '/home/areebadnan/Areeb_code/work/Atheritia/Datasets/47_logos_dataset/10_classes_final/final/split2/47_(10)_cleaned_images/labels'
    class_names = ['1', '3', '4', '5', '9']
    dest_images_folder = 'OCR_detection/Dataset/Textual_logos_dataset/47_(10)_textual_dataset/images'
    dest_labels_folder = 'OCR_detection/Dataset/Textual_logos_dataset/47_(10)_textual_dataset/labels'

    # Copy files based on the class names
    copy_files(images_folder, labels_folder, class_names, dest_images_folder, dest_labels_folder)

    print("Process completed.")