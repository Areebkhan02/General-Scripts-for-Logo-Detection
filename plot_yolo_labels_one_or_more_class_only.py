from tqdm import tqdm
import os
import cv2

def parse_yolo_annotation(label_file, image_width, image_height):
    """
    Parse annotations from a YOLO format label file.

    Args:
        label_file (str): Path to the YOLO format label file.
        image_width (int): Width of the corresponding image.
        image_height (int): Height of the corresponding image.

    Returns:
        list: List of tuples containing annotations (class_id, x1, y1, x2, y2).
    """
    annotations = []
    with open(label_file, 'r') as file:
        lines = file.readlines()
        for line in lines:
            class_id, x_center, y_center, box_width, box_height = map(float, line.strip().split())
            # Convert YOLO format coordinates to absolute pixel coordinates
            x1 = int((x_center - box_width / 2) * image_width)
            y1 = int((y_center - box_height / 2) * image_height)
            x2 = int((x_center + box_width / 2) * image_width)
            y2 = int((y_center + box_height / 2) * image_height)
            annotations.append((int(class_id), x1, y1, x2, y2))
    return annotations

def draw_annotations(image, annotations, class_names):
    """
    Draw bounding box annotations on the image.

    Args:
        image (numpy.ndarray): Input image.
        annotations (list): List of tuples containing annotations (class_id, x1, y1, x2, y2).
        class_names (list): List of class names.
    """
    for annotation in annotations:
        class_id, x1, y1, x2, y2 = annotation
        # Draw bounding box in red
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Calculate text size for background
        label = class_names[class_id]
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        
        # Draw background rectangle for text in red
        cv2.rectangle(image, (x1, y1 - text_height - 10), (x1 + text_width, y1), (0, 255, 0), -1)
        
        # Add class label in white
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

def main(image_folder, label_folder, class_names, output_folder, filter_classes):
    """
    Main function to process images and draw annotations.

    Args:
        image_folder (str): Path to the folder containing images.
        label_folder (str): Path to the folder containing label files.
        class_names (list): List of class names.
        output_folder (str): Path to the output folder where annotated images will be saved.
        filter_classes (list): List of class names to filter and plot.
    """
    # Get class IDs for the specified filter classes
    filter_class_ids = [class_names.index(cls) for cls in filter_classes]
    
    # List image files
    image_files = [file for file in os.listdir(image_folder) if file.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # Iterate over all image files
    for image_file in tqdm(image_files, desc='Processing images'):
        # Load image
        image_path = os.path.join(image_folder, image_file)
        image = cv2.imread(image_path)
        image_height, image_width, _ = image.shape

        # Load corresponding label file
        label_file = os.path.join(label_folder, os.path.splitext(image_file)[0] + '.txt')
        if os.path.exists(label_file):
            # Parse annotations from label file
            annotations = parse_yolo_annotation(label_file, image_width, image_height)
            # Filter annotations for the specified classes only
            filtered_annotations = [anno for anno in annotations if anno[0] in filter_class_ids]
            if filtered_annotations:
                # Draw annotations on image
                draw_annotations(image, filtered_annotations, class_names)
                # Save annotated image to output folder
                output_image_path = os.path.join(output_folder, os.path.splitext(image_file)[0] + '_annotated.jpg')
                cv2.imwrite(output_image_path, image)

if __name__ == '__main__':
    # Input folders and output folder
    image_folder = '/home/areebadnan/Areeb_code/work/Visua_Data/output_videos/64b8705843ad690f994e452e/images'  # Path to the folder containing images
    label_folder = '/home/areebadnan/Areeb_code/work/Visua_Data/output_videos/64b8705843ad690f994e452e/labels'  # Path to the folder containing label files
    output_folder = '/home/areebadnan/Areeb_code/work/Atheritia/output'  # Path to the output folder where annotated images will be saved
    os.makedirs(output_folder, exist_ok=True)

    # Class names
    class_names = ['Audi', 'Mercedes', 'Toyota', 'Porsche', 'Nike', 'Adidas', 'Fly-Emirates',
                   'Hummel', 'Coca-Cola', 'Qatar', 'T-Mobile', 'Allianz', 'Magenta-Sport',
                   'bwin', 'DEVK', 'RheinEnergie', 'Rewe']

    # Classes to filter and plot
    filter_classes = ['Coca-Cola']  # Example: User can specify the classes here

    # Process images and draw annotations
    main(image_folder, label_folder, class_names, output_folder, filter_classes)
