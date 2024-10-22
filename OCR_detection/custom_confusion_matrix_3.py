import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import linear_sum_assignment

# Define the Intersection over Union (IoU) threshold
IOU_THRESHOLD = 0.15

def compute_iou(groundtruth_box, detection_box):
    """
    Compute the Intersection over Union (IoU) between two bounding boxes.

    Args:
        groundtruth_box (list): Bounding box in the format [ymin, xmin, ymax, xmax].
        detection_box (list): Bounding box in the format [ymin, xmin, ymax, xmax].

    Returns:
        float: IoU value between the two bounding boxes.
    """
    # Unpack the coordinates of the ground truth and detection boxes
    g_ymin, g_xmin, g_ymax, g_xmax = tuple(groundtruth_box)
    d_ymin, d_xmin, d_ymax, d_xmax = tuple(detection_box)
    
    # Determine the coordinates of the intersection rectangle
    xa = max(g_xmin, d_xmin)
    ya = max(g_ymin, d_ymin)
    xb = min(g_xmax, d_xmax)
    yb = min(g_ymax, d_ymax)

    # Compute the area of the intersection
    intersection = max(0, xb - xa) * max(0, yb - ya)

    # Compute the area of both the ground truth and detection boxes
    box_a_area = (g_xmax - g_xmin) * (g_ymax - g_ymin)
    box_b_area = (d_xmax - d_xmin) * (d_ymax - d_ymin)

    # Calculate IoU by dividing the intersection area by the union area
    return intersection / float(box_a_area + box_b_area - intersection)

def convert_yolo_format_to_bbox(yolo_bbox):
    """
    Convert YOLO format (x_center, y_center, width, height) to bounding box format [ymin, xmin, ymax, xmax].

    Args:
        yolo_bbox (list): Bounding box in YOLO format.

    Returns:
        list: Bounding box in the format [ymin, xmin, ymax, xmax].
    """
    # Unpack YOLO format bounding box parameters
    x_center, y_center, width, height = yolo_bbox
    
    # Convert YOLO format to standard bounding box format
    xmin = x_center - width / 2
    ymin = y_center - height / 2
    xmax = x_center + width / 2
    ymax = y_center + height / 2
    
    return [ymin, xmin, ymax, xmax]

def load_labels_from_file(file_path):
    """
    Load labels from a YOLOv8 format text file.

    Args:
        file_path (str): Path to the label file.

    Returns:
        list: List of labels with class ID and bounding box coordinates.
    """
    labels = []
    
    # Open and read the label file
    with open(file_path, 'r') as file:
        for line in file:
            # Parse each line to extract class ID and bounding box
            parts = list(map(float, line.strip().split()))
            class_id = int(parts[0])
            bbox = convert_yolo_format_to_bbox(parts[1:])
            labels.append([class_id] + bbox)
    
    return labels

def process_detections(groundtruth_dir, detection_dir, categories):
    """
    Process ground truth and detection files to create a confusion matrix using the Hungarian Algorithm.

    Args:
        groundtruth_dir (str): Directory containing ground truth label files.
        detection_dir (str): Directory containing detection label files.
        categories (list): List of category dictionaries with 'id' and 'name' keys.

    Returns:
        np.ndarray: Confusion matrix of shape (num_classes + 1, num_classes + 1).
    """
    num_classes = len(categories)
    # Initialize confusion matrix with shape (predicted_classes + 1, true_classes + 1)
    confusion_matrix = np.zeros(shape=(num_classes + 1, num_classes + 1), dtype=int)
    image_files = [f for f in os.listdir(groundtruth_dir) if f.endswith('.txt')]
    print(f"Processing {len(image_files)} images...")

    for image_file in image_files:
        groundtruth_boxes = load_labels_from_file(os.path.join(groundtruth_dir, image_file))
        detection_boxes = load_labels_from_file(os.path.join(detection_dir, image_file))

        if len(groundtruth_boxes) == 0 and len(detection_boxes) == 0:
            continue

        # Create cost matrix based on IoU
        iou_matrix = np.zeros((len(detection_boxes), len(groundtruth_boxes)), dtype=float)

        for dt_idx, dt_box in enumerate(detection_boxes):
            for gt_idx, gt_box in enumerate(groundtruth_boxes):
                iou = compute_iou(gt_box[1:], dt_box[1:])
                if iou >= IOU_THRESHOLD:
                    # We use negative IoU because linear_sum_assignment minimizes the total cost
                    iou_matrix[dt_idx, gt_idx] = -iou
                else:
                    iou_matrix[dt_idx, gt_idx] = 1e6  # A large cost for non-matching

        # Perform Hungarian Matching
        if iou_matrix.size > 0:
            dt_indices, gt_indices = linear_sum_assignment(iou_matrix)
        else:
            dt_indices, gt_indices = np.array([]), np.array([])

        matched_dt = set()
        matched_gt = set()

        for dt_idx, gt_idx in zip(dt_indices, gt_indices):
            if iou_matrix[dt_idx, gt_idx] < 1e6:
                dt_class = int(detection_boxes[dt_idx][0])
                gt_class = int(groundtruth_boxes[gt_idx][0])
                confusion_matrix[dt_class][gt_class] += 1
                matched_dt.add(dt_idx)
                matched_gt.add(gt_idx)

        # Handle False Negatives (FN): Ground truth objects not detected
        for gt_idx, gt_box in enumerate(groundtruth_boxes):
            if gt_idx not in matched_gt:
                true_class = int(gt_box[0])
                confusion_matrix[-1][true_class] += 1  # Extra Row for FN

        # Handle False Positives (FP): Detections that do not match any ground truth
        for dt_idx, dt_box in enumerate(detection_boxes):
            if dt_idx not in matched_dt:
                pred_class = int(dt_box[0])
                confusion_matrix[pred_class][-1] += 1  # Extra Column for FP

    print("Processing completed.")
    return confusion_matrix

def save_confusion_matrix(confusion_matrix, class_names, output_path, normalized=False):
    """
    Save the confusion matrix as an image file.

    Args:
        confusion_matrix (np.ndarray): Confusion matrix to be saved.
        class_names (list): List of class names.
        output_path (str): Directory where the image will be saved.
        normalized (bool): If True, normalize the confusion matrix.
    """
    plt.figure(figsize=(14, 12))
    
    if normalized:
        # Normalize the confusion matrix by dividing each column by the sum of the column
        column_sums = confusion_matrix.sum(axis=0)
        # To avoid division by zero
        column_sums[column_sums == 0] = 1
        confusion_matrix_normalized = confusion_matrix.astype('float') / column_sums
        title = 'Normalized Confusion Matrix'
    else:
        confusion_matrix_normalized = confusion_matrix
        title = 'Confusion Matrix'
    
    # Determine format for displaying matrix values
    fmt = ".2f" if np.issubdtype(confusion_matrix_normalized.dtype, np.floating) else "d"
    
    # Create an annotation matrix where zeros are replaced with empty strings
    annot_matrix = np.where(
        confusion_matrix_normalized == 0,
        "0",
        np.round(confusion_matrix_normalized, 2).astype(str)
    )
    
    # **Updated Labels:**
    # - Rows: Predicted labels + 'False Negatives'
    # - Columns: True labels + 'False Positives'
    sns.heatmap(confusion_matrix_normalized, annot=annot_matrix, fmt="", cmap="Blues", 
                xticklabels=class_names + ['False Positive'], 
                yticklabels=class_names + ['False Negative'],
                cbar_kws={'label': 'Normalized Value'})
    
    # Set the title and labels for the confusion matrix plot
    plt.title(title, fontsize=16)
    plt.xlabel('True Labels', fontsize=14)
    plt.ylabel('Predicted Labels', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Save the confusion matrix plot to a file
    file_suffix = '_normalized' if normalized else ''
    file_name = f'confusion_matrix{file_suffix}.png'
    plt.savefig(os.path.join(output_path, file_name), dpi=300)
    plt.close()

def display(confusion_matrix, categories, output_path):
    """
    Display the confusion matrix and calculate precision and recall for each category.

    Args:
        confusion_matrix (np.ndarray): Confusion matrix.
        categories (list): List of category dictionaries with 'id' and 'name' keys.
        output_path (str): Directory where metrics and matrices will be saved.
    """
    print("\nConfusion Matrix:")
    print(confusion_matrix, "\n")
    results = []

    # Calculate precision and recall for each category
    for i in range(len(categories)):
        pred_class = categories[i]["id"]
        name = categories[i]["name"]
        
        # True Positives: correctly predicted instances
        TP = confusion_matrix[pred_class][pred_class]
        # False Positives: predictions that don't match true labels
        FP = confusion_matrix[pred_class][-1]
        # False Negatives: true labels that weren't predicted
        FN = confusion_matrix[-1][pred_class]
        
        # Precision: TP / (TP + FP)
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        # Recall: TP / (TP + FN)
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        
        results.append({
            'category': name, 
            f'precision_@{IOU_THRESHOLD}IOU': f'{precision:.2f}', 
            f'recall_@{IOU_THRESHOLD}IOU': f'{recall:.2f}'
        })
    
    # Create a DataFrame from the results and save it to a CSV file
    df = pd.DataFrame(results)
    print(df)
    df.to_csv(os.path.join(output_path, 'iou_metrics.csv'), index=False)
    
    # Save both normalized and unnormalized confusion matrices
    class_names = [category['name'] for category in categories]
    save_confusion_matrix(confusion_matrix, class_names, output_path, normalized=False)
    save_confusion_matrix(confusion_matrix, class_names, output_path, normalized=True)

# Define directories and categories
groundtruth_dir = 'Dataset/Textual_logos_dataset/17_logo_textual_dataset/labels'
detection_dir = 'Dataset/Textual_logos_dataset/17_logo_textual_dataset/new_try/ocr_labels'
output_path = 'Dataset/Textual_logos_dataset/17_logo_textual_dataset/new_try/output_0.15'

categories = [
    {"id": 0, "name": "adidas"},
    {"id": 1, "name": "coca-cola"},
    {"id": 2, "name": "qatar"},
    {"id": 3, "name": "allianz"},
    {"id": 4, "name": "bwin"},
    {"id": 5, "name": "devk"},
    {"id": 6, "name": "rheinenergie"},
    {"id": 7, "name": "rewe"}
]


# categories = [
#     {"id": 0, "name": "Stahlwerk"},
#     {"id": 1, "name": "tipico"},
#     {"id": 2, "name": "betway"},
#     {"id": 3, "name": "sap"},
#     {"id": 4, "name": "penny"}
# ]


# categories = [
#     {"id": 0, "name": "Paulaner"},
#     {"id": 1, "name": "BKW"},
#     {"id": 2, "name": "Konami"},
#     {"id": 3, "name": "Gaffel"},
#     {"id": 4, "name": "DHL"},
#     {"id": 5, "name": "E Football"},
#     {"id": 6, "name": "Bitburger"},
#     {"id": 7, "name": "Viessmann"},
#     {"id": 8, "name": "Helvetia"},
#     {"id": 9, "name": "Union Investment"},
#     {"id": 10, "name": "Raiffeisen"},
#     {"id": 11, "name": "Libertex"},
#     {"id": 12, "name": "ERGO"},
#     {"id": 13, "name": "Wiesenhof"},
#     {"id": 14, "name": "Siemens"},
#     {"id": 15, "name": "EWE"},
#     {"id": 16, "name": "PreZero"},
#     {"id": 17, "name": "Einhell"},
#     {"id": 18, "name": "Adobe"},
#     {"id": 19, "name": "Autohero"},
#     {"id": 20, "name": "Henkel"},
#     {"id": 21, "name": "Flyeralarm"},
#     {"id": 22, "name": "Sunrise"},
#     {"id": 23, "name": "Hylo"},
#     {"id": 24, "name": "94,3 rs2"}
# ]
# Example usage:
confusion_matrix = process_detections(groundtruth_dir, detection_dir, categories)
display(confusion_matrix, categories, output_path)
