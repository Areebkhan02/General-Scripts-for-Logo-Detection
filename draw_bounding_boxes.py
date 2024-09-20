import os
import cv2

# Paths to your directories
image_dir = 'Datasets/3heads_merged_dataset/47_37_val_logos_dataset/issue_checking/ERGO/images'  # Update with your image directory
label_dir = 'Datasets/3heads_merged_dataset/47_37_val_logos_dataset/issue_checking/ERGO/labels'  # Update with your label directory
output_dir = 'Datasets/3heads_merged_dataset/47_37_val_logos_dataset/issue_checking/ERGO/bbox'  # Update with your output directory

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Define a mapping from class ID to class name
class_mapping = {
    4: 'BetWay',  # Replace with your actual class names
}

# Function to load YOLO formatted labels
def load_yolo_labels(label_path):
    bboxes = []
    with open(label_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            class_id, x_center, y_center, width, height = map(float, line.strip().split())
            bboxes.append((int(class_id), x_center, y_center, width, height))
    return bboxes

# Function to draw bounding boxes on images
def draw_bboxes(image, bboxes):
    h, w, _ = image.shape
    for bbox in bboxes:
        class_id, x_center, y_center, width, height = bbox
        
        # Convert YOLO format to top-left and bottom-right corners
        x1 = int((x_center - width / 2) * w)
        y1 = int((y_center - height / 2) * h)
        x2 = int((x_center + width / 2) * w)
        y2 = int((y_center + height / 2) * h)
        
        # Get the class name from the mapping
        class_name = class_mapping.get(class_id, f'Class {class_id}')
        #print(class_name)
        
        # Draw rectangle and put label
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
    return image

# Iterate through images and corresponding labels
for filename in os.listdir(image_dir):
    if filename.endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(image_dir, filename)
        label_path = os.path.join(label_dir, os.path.splitext(filename)[0] + '.txt')
        
        if os.path.exists(label_path):
            image = cv2.imread(image_path)
            bboxes = load_yolo_labels(label_path)
            image_with_bboxes = draw_bboxes(image, bboxes)
            
            # Save the output image
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, image_with_bboxes)
        else:
            print(f"Label not found for {filename}")

print("Bounding boxes with class names have been drawn and images saved.")
