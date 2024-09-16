import cv2
import numpy as np
import random
import os
from tqdm import tqdm
import math

def resize_logo(logo, background_img_size, min_size=60, max_size=100):
    """ 
    Resize the logo to a random size between min_size and max_size.
    """
    size = random.randint(min_size, max_size)
    size = size * math.ceil(background_img_size/1000)
    logo_resized = cv2.resize(logo, (size, size), interpolation=cv2.INTER_AREA)
    return logo_resized

def paste_logo(background_img, logo_crop, position, annotations, class_label):
    """
    Paste a logo onto the background image at the specified position and update annotations.
    """
    # Get dimensions of the logo crop
    logo_h, logo_w, _ = logo_crop.shape

    # Extract the region of interest from the background image
    x, y = position
    roi = background_img[y:y + logo_h, x:x + logo_w]

    # Paste the logo onto the background image without alpha blending
    np.copyto(roi, logo_crop)

    # Calculate the YOLO coordinates of the pasted logo
    bg_h, bg_w, _ = background_img.shape
    x_center = (x + logo_w / 2) / bg_w
    y_center = (y + logo_h / 2) / bg_h
    width = logo_w / bg_w
    height = logo_h / bg_h

    # Add the YOLO coordinates to the annotations list
    annotations.append((class_label, x_center, y_center, width, height))

def check_overlap(x, y, logo_w, logo_h, placed_logos):
    """
    Check if the new logo overlaps with any already placed logos.
    """
    for px, py, pw, ph in placed_logos:
        if not (x + logo_w <= px or x >= px + pw or y + logo_h <= py or y >= py + ph):
            return True
    return False

def main():
    # Paths to background images and logos
    background_imgs_path = "/home/jansher/Desktop/bg_img/bg_images"
    logos_base_path = '/home/jansher/Desktop/bg_img/logo'
    output_images_path = '/home/jansher/Desktop/bg_img/images'
    output_labels_path = '/home/jansher/Desktop/bg_img/labels'

    # Create output directories if they do not exist
    os.makedirs(output_images_path, exist_ok=True)
    os.makedirs(output_labels_path, exist_ok=True)

    # Map class labels to folder names
    class_labels = {
        0: 'Audi', 1: 'Mercedes', 2: 'Toyota', 3: 'Porsche', 4: 'Nike',
        5: 'Adidas', 6: 'Fly-Emirates', 7: 'Hummel', 8: 'Coca-Cola',
        9: 'Qatar', 10: 'T-Mobile', 11: 'Allianz', 12: 'Magenta-Sport',
        13: 'bwin', 14: 'DEVK', 15: 'RheinEnergie', 16: 'Rewe'
    }

    # Loop through each folder containing different types of logos
    for class_label, logo_folder in class_labels.items():
        logos_path = os.path.join(logos_base_path, logo_folder)
        # Check if the logo folder exists
        if not os.path.exists(logos_path):
            print(f"Logo folder {logo_folder} does not exist. Skipping processing for class label {class_label}.")
            continue

        # Load all logos from the current folder
        logos = [os.path.join(logos_path, f) for f in os.listdir(logos_path) if f.endswith(('.jpg', '.jpeg', '.png'))]

        # Define the number of logos to paste on each background image
        num_logos = min(15, len(logos))  # Paste at most 15 logos per background image

        # Process each background image
        for bg_img_file in tqdm(os.listdir(background_imgs_path)):
            bg_img_path = os.path.join(background_imgs_path, bg_img_file)
            background_img = cv2.imread(bg_img_path)
            background_img_size = background_img.shape[0]
            if background_img is None:
                print(f"Error: Could not load background image from {bg_img_path}")
                continue

            # Initialize list to store YOLO annotations and placed logos
            annotations = []
            placed_logos = []

            # Paste logos onto background image without overlapping
            for _ in range(num_logos):
                logo_crop_file = random.choice(logos)
                logo_crop = cv2.imread(logo_crop_file)
                if logo_crop is None:
                    print(f"Error: Could not load logo image from {logo_crop_file}")
                    continue

                logo_crop = resize_logo(logo_crop, background_img_size)

                for attempt in range(100):  # Try up to 100 times to find a non-overlapping position
                    # Randomly select position to paste logo
                    x = random.randint(0, background_img.shape[1] - logo_crop.shape[1])
                    y = random.randint(0, background_img.shape[0] - logo_crop.shape[0])

                    # Check for overlap with previously placed logos
                    if not check_overlap(x, y, logo_crop.shape[1], logo_crop.shape[0], placed_logos):
                        position = (x, y)

                        # Paste logo onto background image and save YOLO coordinates
                        paste_logo(background_img, logo_crop, position, annotations, class_label)

                        # Add the logo's position and dimensions to the placed_logos list
                        placed_logos.append((x, y, logo_crop.shape[1], logo_crop.shape[0]))
                        break
                else:
                    print(f"Warning: Could not find non-overlapping position for logo {_ + 1}")

            # Save augmented image and corresponding annotations
            base_name = os.path.splitext(bg_img_file)[0]
            augmented_img_path = os.path.join(output_images_path, f"{base_name}_{logo_folder}_aug.jpg")
            annotations_path = os.path.join(output_labels_path, f"{base_name}_{logo_folder}_aug.txt")

            cv2.imwrite(augmented_img_path, background_img)

            with open(annotations_path, 'w') as f:
                for annotation in annotations:
                    f.write(' '.join(map(str, annotation)) + '\n')

if __name__ == "__main__":
    main()
