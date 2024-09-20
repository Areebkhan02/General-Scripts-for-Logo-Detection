import os
import shutil
from tqdm import tqdm

def copy_files(images_folder, labels_folder, class_name, dest_images_folder, dest_labels_folder):
    """
    Copies text files and their corresponding images to new folders based on a given class name.

    Args:
        images_folder (str): Path to the folder containing images.
        labels_folder (str): Path to the folder containing YOLO label files.
        class_name (str): The class name to search for in the YOLO label files.
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

            # Check if the class name exists in the label file
            class_found = False
            for line in lines:
                line_parts = line.strip().split()
                if len(line_parts) > 0 and line_parts[0] == class_name:
                    class_found = True
                    break

            # If the class is found, copy the label file and corresponding image
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
    images_folder = 'Datasets/3heads_merged_dataset/37_logos_train_data/images'
    labels_folder = 'Datasets/3heads_merged_dataset/37_logos_train_data/labels'
    class_name = '31'                                                                                                                                                                                                                                                                                                                                                                               
    dest_images_folder = 'Datasets/3heads_merged_dataset/37_logos_train_data/issues/Henkel/images'
    dest_labels_folder = 'Datasets/3heads_merged_dataset/37_logos_train_data/issues/Henkel/labels'

    # Copy files based on the class name
    copy_files(images_folder, labels_folder, class_name, dest_images_folder, dest_labels_folder)

    print("Process completed.")
