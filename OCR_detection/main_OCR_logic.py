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
    logos_dict = {
    'paulaner': 0,
    'bkw': 1,
    'konami': 2,
    'gaffel': 3,
    'dhl': 4,
    'e football': 5,
    'bitburger': 6,
    'viessmann': 7,
    'helvetia': 8,
    'union investment': 9,
    'raiffeisen': 10,
    'libertex': 11,
    'ergo': 12,
    'wiesenhof': 13,
    'siemens': 14,
    'ewe': 15,
    'prezero': 16,
    'einhell': 17,
    'adobe': 18,
    'autohero': 19,
    'henkel': 20,
    'flyeralarm': 21,
    'sunrise': 22,
    'hylo': 23,
    '94,3 rs2': 24
}
    
    # Process each image in the input folder
    for image_filename in os.listdir(input_folder):
        if image_filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_folder, image_filename)
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

def main():
    input_folder = 'OCR_detection/Dataset/Textual_logos_dataset/37_logo_cleaned_data/37_textual_logos/images'
    output_image_folder = 'OCR_detection/Dataset/Textual_logos_dataset/37_logo_cleaned_data/37_textual_logos/ocr_images'
    output_label_folder = 'OCR_detection/Dataset/Textual_logos_dataset/37_logo_cleaned_data/37_textual_logos/ocr_labels'
    
    # Ensure output directories exist
    os.makedirs(output_image_folder, exist_ok=True)
    os.makedirs(output_label_folder, exist_ok=True)
    
    process_images(input_folder, output_image_folder, output_label_folder)

if __name__ == "__main__":
    main()
