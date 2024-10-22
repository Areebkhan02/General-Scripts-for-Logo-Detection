import os
import cv2
import numpy as np
from paddleocr import PaddleOCR, draw_ocr
from difflib import get_close_matches

def initialize_ocr():
    return PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False, verbose=False)

def perform_ocr(ocr, image_path):
    return ocr.ocr(image_path, rec=True)

def filter_ocr_results(result, logos_dict):
    if result is None or not result[0]:
        print("No text detected in the image.")
        return [[]]  # Return an empty list to indicate no results

    filtered_results = [[]]
    for line in result[0]:
        bbox, (text, confidence) = line
        processed_text = text.lower().strip()
        if processed_text in logos_dict:
            filtered_results[0].append((bbox, (logos_dict[processed_text], confidence)))
        else:
            best_match = get_close_matches(processed_text, logos_dict.keys(), n=1, cutoff=0.6)
            if best_match:
                filtered_results[0].append((bbox, (logos_dict[best_match[0]], confidence)))
    return filtered_results

def convert_to_yolo_format(bbox, image_width, image_height):
    box = np.array(bbox).astype(np.float32)
    xmin = min(box[:, 0])
    ymin = min(box[:, 1])
    xmax = max(box[:, 0])
    ymax = max(box[:, 1])
    
    x_center = (xmin + xmax) / 2.0
    y_center = (ymin + ymax) / 2.0
    width = xmax - xmin
    height = ymax - ymin
    
    x_center /= image_width
    y_center /= image_height
    width /= image_width
    height /= image_height
    
    return [x_center, y_center, width, height]

def plot_yolo_labels(image, yolo_bboxes, image_width, image_height, logos_dict):
    # Reverse the logos_dict to map class numbers back to logo names
    class_to_logo = {v: k for k, v in logos_dict.items()}
    
    for class_number, bbox in yolo_bboxes:
        x_center, y_center, width, height = bbox
        x_center *= image_width
        y_center *= image_height
        width *= image_width
        height *= image_height
        
        xmin = int(x_center - width / 2)
        ymin = int(y_center - height / 2)
        xmax = int(x_center + width / 2)
        ymax = int(y_center + height / 2)
        
        # Get the logo name from the class number
        text = class_to_logo.get(class_number, "Unknown")
        
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(image, text, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

def save_labels_with_bboxes(yolo_bboxes, label_file_path):
    with open(label_file_path, 'w') as f:
        for class_number, bbox in yolo_bboxes:
            f.write(f"{class_number} {bbox[0]:.5f} {bbox[1]:.5f} {bbox[2]:.5f} {bbox[3]:.5f}\n")

def process_images(input_folder, output_image_folder, output_label_folder):
    # Initialize PaddleOCR
    ocr = initialize_ocr()
    
    # Dictionary of logos with class numbers
    # logos_dict = {
    #     'Stahlwerk': 0,
    #     'tipico': 1,
    #     'betway': 2,
    #     'sap': 3,
    #     'penny': 4
    # }

    logos_dict = {
        'adidas': 0,
        'coca-cola': 1,
        'qatar': 2,
        'allianz': 3,
        'bwin': 4,
        'devk': 5,
        'rheinenergie': 6,
        'rewe': 7
    }
    
    # Process each image in the input folder
    for image_filename in os.listdir(input_folder):
        if image_filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_folder, image_filename)
            # Print the filename before performing OCR
            print(f"Performing OCR on: {image_filename}")
            image = cv2.imread(image_path)
            image_height, image_width = image.shape[:2]
            
            # Perform OCR
            result = perform_ocr(ocr, image_path)
            
            # Filter OCR results
            filtered_results = filter_ocr_results(result, logos_dict)
            
            # Convert bounding boxes to YOLO format and save labels
            yolo_bboxes = []
            for line in filtered_results[0]:
                bbox, (class_number, confidence) = line
                yolo_bbox = convert_to_yolo_format(bbox, image_width, image_height)
                yolo_bboxes.append((class_number, yolo_bbox))
            
            # Plot YOLO labels on the image
            plot_yolo_labels(image, yolo_bboxes, image_width, image_height, logos_dict)
            
            # Save the annotated image
            annotated_image_path = os.path.join(output_image_folder, f"{os.path.splitext(image_filename)[0]}_annotated.jpg")
            cv2.imwrite(annotated_image_path, image)
            print(f"Annotated image saved at {annotated_image_path}")
            
            # Save the labels and bounding boxes to a text file
            label_file_path = os.path.join(output_label_folder, f"{os.path.splitext(image_filename)[0]}.txt")
            save_labels_with_bboxes(yolo_bboxes, label_file_path)
            print(f"Labels and bounding boxes saved at {label_file_path}")

            del image
            cv2.destroyAllWindows()

def main():
    input_folder = 'Dataset/Textual_logos_dataset/17_logo_textual_dataset/images'
    output_image_folder = 'Dataset/Textual_logos_dataset/17_logo_textual_dataset/new_try/ocr_images'
    output_label_folder = 'Dataset/Textual_logos_dataset/17_logo_textual_dataset/new_try/ocr_labels'
    
    # Ensure output directories exist
    os.makedirs(output_image_folder, exist_ok=True)
    os.makedirs(output_label_folder, exist_ok=True)
    
    process_images(input_folder, output_image_folder, output_label_folder)

if __name__ == "__main__":
    main()
