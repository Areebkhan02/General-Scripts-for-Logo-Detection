import os
import random
import cv2
import numpy as np
from tqdm import tqdm

def plot_bounding_boxes(images_folder, labels_folder, save_path, num_images):
    """
    Randomly selects a specified number of images from the images folder, plots the bounding boxes from the 
    respective YOLO label files, and saves the processed images in a separate folder.

    Args:
        images_folder (str): Path to the folder containing the images.
        labels_folder (str): Path to the folder containing the YOLO format label files.
        save_path (str): Path to the folder where processed images will be saved.
        num_images (int): Number of images to process.
    """
    # Create the save directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    # Get the list of image files
    image_files = os.listdir(images_folder)
    
    # Randomly select the specified number of images
    selected_images = random.sample(image_files, num_images)

    for image_file in tqdm(selected_images, desc="Processing images"):
        # Get the corresponding label file
        label_file = os.path.splitext(image_file)[0] + ".txt"
        label_path = os.path.join(labels_folder, label_file)

        # Open the image using OpenCV
        image_path = os.path.join(images_folder, image_file)
        image = cv2.imread(image_path)

        # Read the label file and plot the bounding boxes
        if os.path.exists(label_path):
            with open(label_path, "r") as file:
                for line in file:
                    parts = line.strip().split()
                    class_id = parts[0]
                    x_center, y_center, width, height = map(float, parts[1:])

                    # Convert YOLO format to bounding box coordinates
                    img_height, img_width = image.shape[:2]
                    x_min = int((x_center - width / 2) * img_width)
                    y_min = int((y_center - height / 2) * img_height)
                    x_max = int((x_center + width / 2) * img_width)
                    y_max = int((y_center + height / 2) * img_height)

                    # Draw the bounding box
                    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color=(0, 0, 255), thickness=6)

                    # Set font parameters
                    font_scale = 0.7
                    font_thickness = 2
                    font = cv2.FONT_HERSHEY_SIMPLEX

                    # Get text size
                    (text_width, text_height), baseline = cv2.getTextSize(class_id, font, font_scale, font_thickness)
                    
                    # Draw label background and text
                    label_background = (0, 0, 255)
                    label_text_color = (255, 255, 255)
                    cv2.rectangle(image, (x_min, y_min - text_height - baseline), (x_min + text_width, y_min), label_background, thickness=cv2.FILLED)
                    cv2.putText(image, class_id, (x_min, y_min - baseline), font, font_scale, label_text_color, font_thickness, lineType=cv2.LINE_AA)

        # Save the processed image
        save_image_path = os.path.join(save_path, image_file)
        cv2.imwrite(save_image_path, image)

    print(f"Processed images saved in {save_path}")

# Example usage:
if __name__ == '__main__':
    images_folder = "Datasets/47_logos_dataset/10_classes_final/final/split/train/combined_aug_org/images"
    labels_folder = "Datasets/47_logos_dataset/10_classes_final/final/split/train/combined_aug_org/labels"
    save_path = "Datasets/47_logos_dataset/10_classes_final/final/split/train/combined_aug_org/bbox_test"
    num_images = 50

    plot_bounding_boxes(images_folder, labels_folder, save_path, num_images)
