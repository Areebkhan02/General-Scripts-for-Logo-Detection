import os
import numpy as np
import cv2
from paddleocr import PaddleOCR
from difflib import get_close_matches

# Initialize PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False, verbose=False)

# List of logos # logos in lowercase
logos = ['herthabsc', 'autohero', 'hyundai', 'coca-cola', 'betway', '94/rs2', 'berlinerkindl', 'homeday', 'rewe', 'ford', 'hummel', 'stahlwerk', 'bwin', 'magentasport']

def process_image(img_path):
    """
    Run OCR on a single image and process the results.
    """
    result = ocr.ocr(img_path, rec=True)
    original_results = [[]]  # Initialize the list to store results in the same format as `result`

    for line in result[0]:
        bbox, (text, confidence) = line
        processed_text = text.lower().strip()
        
        if processed_text in logos:
            original_results[0].append((bbox, (text, confidence)))
        else:
            best_match = get_close_matches(processed_text, logos, n=1, cutoff=0.6)
            if best_match:
                original_results[0].append((bbox, (best_match[0], confidence)))

    return original_results

def draw_boxes(image, result):
    """
    Draw bounding boxes and labels on the image.
    """
    for line in result[0]:
        bbox, (text, confidence) = line
        bbox = np.array(bbox, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(image, [bbox], isClosed=True, color=(0, 0, 255), thickness=3)

        text_position = (bbox[0][0][0], bbox[0][0][1] - 10)
        (text_width, text_height), _ = cv2.getTextSize(f"{text} ({confidence:.2f})", 
                                                         cv2.FONT_HERSHEY_SIMPLEX, 
                                                         0.7, 2)
        
        rectangle_start = (text_position[0], text_position[1] - text_height - 10)
        rectangle_end = (text_position[0] + text_width, text_position[1])
        cv2.rectangle(image, rectangle_start, rectangle_end, (255, 0, 0), thickness=cv2.FILLED)

        cv2.putText(image, f"{text} ({confidence:.2f})", text_position,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    return image

if __name__ == "__main__":
    input_folder = 'input_images/testing_images_new'
    output_folder = 'output_images/output_images_test'

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(input_folder, filename)
            original_results = process_image(img_path)

            image = cv2.imread(img_path)
            image_with_boxes = draw_boxes(image.copy(), original_results)

            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, image_with_boxes)