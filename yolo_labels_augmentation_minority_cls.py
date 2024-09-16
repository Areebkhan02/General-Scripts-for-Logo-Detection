import os
import cv2
import numpy as np
import albumentations as A
from tqdm import tqdm

# Define augmentations
augmentation = A.Compose([
    #A.RandomBrightnessContrast(p=0.8),
    #A.HueSaturationValue(p=0.5),
    A.Rotate(limit=(45,45), p=1.0),
    A.Rotate(limit=(-65,-45), p=1.0),
    A.Rotate(limit=(46,180), p=1.0)],
    #A.Rotate(limit=(), p=)
    # A.Shear(p=0.5),
    #A.Resize(height=640, width=1080, p=1),
    #A.ChannelShuffle(always_apply=False,p=.5),
    #A.ToGray(p=.2),
    #A.SmallestMaxSize(p=.6),
    #A.RandomSizedBBoxSafeCrop(640, 640, p=0.8),
    # A.ShiftScaleRotate(p=0.3),
    # A.HorizontalFlip(p=0.5),
    # A.VerticalFlip(p=0.5),
    #A.RandomScale(scale_limit=0.1, p=.8),
    bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_area=0, min_visibility=0))



# Helper function to load image and annotations
def load_data(image_path, annotation_path):
    """
    Load image and annotation data.

    Args:
        image_path (str): Path to the image file.
        annotation_path (str): Path to the annotation file.

    Returns:
        img (numpy.ndarray): Loaded image.
        dict: A dictionary with 'class_labels' and 'bboxes'.
    """
    img = cv2.imread(image_path)
    with open(annotation_path, 'r') as f:
        bboxes = []
        class_labels = []
        for line in f.readlines():
            class_id, x_center, y_center, width, height = map(float, line.strip().split())
            bboxes.append([x_center, y_center, width, height])
            class_labels.append(int(class_id))
    return img, {'class_labels': class_labels, 'bboxes': bboxes}

# Helper function to save augmented image and annotation
def save_data(img, ann, filename, image_dir, annotation_dir):
    """
    Save augmented image and annotation data.

    Args:
        img (numpy.ndarray): Augmented image.
        ann (dict): Dictionary containing 'class_labels' and 'bboxes'.
        filename (str): Filename for the saved image and annotation.
        image_dir (str): Directory to save the augmented image.
        annotation_dir (str): Directory to save the augmented annotation.
    """
    img_path = os.path.join(image_dir, filename)
    ann_path = os.path.join(annotation_dir, os.path.splitext(filename)[0] + '.txt')
    cv2.imwrite(img_path, img)
    with open(ann_path, 'w') as f:
        for class_id, bbox in zip(ann['class_labels'], ann['bboxes']):
            line = f"{class_id} {' '.join(map(str, bbox))}\n"
            f.write(line)

# Normalize bounding boxes to ensure they are within [0.0, 1.0]
def normalize_bboxes(bboxes, img_width, img_height):
    """
    Normalize bounding boxes to be within [0.0, 1.0].

    Args:
        bboxes (list): List of bounding boxes.
        img_width (int): Width of the image.
        img_height (int): Height of the image.

    Returns:
        list: List of normalized bounding boxes.
    """
    normalized_bboxes = []
    for bbox in bboxes:
        x_center, y_center, width, height = bbox
        x_center = np.clip(x_center, 0.0, 1.0)
        y_center = np.clip(y_center, 0.0, 1.0)
        width = np.clip(width, 0.0, 1.0)
        height = np.clip(height, 0.0, 1.0)
        normalized_bboxes.append([x_center, y_center, width, height])
    return normalized_bboxes

# Function to augment a single image and annotation
def augment_single_image(img, ann, filename, minority_classes, augmentation_count, image_dir, annotation_dir):
    """
    Augment a single image and its annotations.

    Args:
        img (numpy.ndarray): Image to augment.
        ann (dict): Dictionary containing 'class_labels' and 'bboxes'.
        filename (str): Filename of the image.
        minority_classes (list): List of minority class IDs to augment more.
        augmentation_count (int): Number of times to augment images containing minority classes.
        image_dir (str): Directory to save augmented images.
        annotation_dir (str): Directory to save augmented annotations.
    """
    class_labels = ann['class_labels']
    bboxes = ann['bboxes']
      
    # Normalize bounding boxes
    img_height, img_width = img.shape[:2]
    bboxes = normalize_bboxes(bboxes, img_width, img_height)
    
    # Augment images containing minority classes more times
    if any(cls in minority_classes for cls in class_labels):
        for i in range(augmentation_count):
            try:
                augmented = augmentation(image=img, bboxes=bboxes, class_labels=class_labels)
                aug_filename = f'aug_{i}_{filename}'
                save_data(augmented['image'], {
                    'class_labels': augmented['class_labels'],
                    'bboxes': augmented['bboxes']
                }, aug_filename, image_dir, annotation_dir)
            except ValueError as e:
                print(f"Error augmenting {filename}: {e}")
                continue
    # Augment images not containing minority classes once
    else:
        try:
            augmented = augmentation(image=img, bboxes=bboxes, class_labels=class_labels)
            aug_filename = f'aug_{filename}'
            save_data(augmented['image'], {
                'class_labels': augmented['class_labels'],
                'bboxes': augmented['bboxes']
            }, aug_filename, image_dir, annotation_dir)
        except ValueError as e:
            print(f"Error augmenting {filename}: {e}")

# Define directories
image_dir = 'Visua_Data/augmentation_test/images'
annotation_dir = 'Visua_Data/augmentation_test/labels'
augmented_image_dir = 'Visua_Data/augmentation_test/aug_images'
augmented_annotation_dir = 'Visua_Data/augmentation_test/aug_labels'

# Create directories to save augmented images and annotations if they don't exist
os.makedirs(augmented_image_dir, exist_ok=True)
os.makedirs(augmented_annotation_dir, exist_ok=True)

# Define minority classes and augmentation count
minority_classes = []  # Example class IDs for minority classes
augmentation_count = 5 # Number of times to augment images containing minority classes

# Process each image and annotation
for filename in tqdm(os.listdir(image_dir), desc="Processing images"):
    if filename.endswith(('.jpg', '.jpeg', '.png')):
        img_path = os.path.join(image_dir, filename)
        ann_path = os.path.join(annotation_dir, os.path.splitext(filename)[0] + '.txt')
        if os.path.exists(ann_path):
            img, ann = load_data(img_path, ann_path)
            augment_single_image(img, ann, filename, minority_classes, augmentation_count, augmented_image_dir, augmented_annotation_dir)
