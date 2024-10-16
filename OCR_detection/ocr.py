# ========================== for multiple images ==========================

# import os
# import numpy as np
# import cv2
# from paddleocr import PaddleOCR
# from pprint import pprint

# # Initialize PaddleOCR
# ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False, verbose=False)  # Use English as an example

# # Specify the folder containing images
# input_folder = 'input_test_images'  # Replace with your folder path
# output_folder = 'output_test_images'  # Replace with your output folder path

# # Create output folder if it doesn't exist
# os.makedirs(output_folder, exist_ok=True)

# # Function to draw bounding boxes and labels
# def draw_boxes(image, result):
#     for line in result[0]:  # Process each detected box and label
#         bbox, (text, confidence) = line
#         bbox = np.array(bbox, dtype=np.int32).reshape((-1, 1, 2))  # Convert bbox points to the correct format

#         # Draw the bounding box on the image
#         cv2.polylines(image, [bbox], isClosed=True, color=(0, 0, 255), thickness=4)

#         # Calculate the position for the text
#         text_position = (bbox[0][0][0], bbox[0][0][1] - 10)

#         # Get the size of the text to create a rectangle background
#         (text_width, text_height), _ = cv2.getTextSize(f"{text} ({confidence:.2f})", 
#                                                          cv2.FONT_HERSHEY_SIMPLEX, 
#                                                          0.7, 2)

#         # # Draw a blue rectangle behind the text
#         # rectangle_start = (text_position[0], text_position[1] - text_height - 10)
#         # rectangle_end = (text_position[0] + text_width, text_position[1])
#         # cv2.rectangle(image, rectangle_start, rectangle_end, (255, 0, 0), thickness=cv2.FILLED)

#         # Place the label with confidence score above the bounding box
#         cv2.putText(image, f"{text} ({confidence:.2f})", text_position,
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2, cv2.LINE_AA)

#     return image

# # Process each image in the input folder
# for filename in os.listdir(input_folder):
#     if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):  # Check for image file types
#         img_path = os.path.join(input_folder, filename)
        
#         # Perform OCR on the image
#         result = ocr.ocr(img_path, rec=True)
        
#         # Load the image
#         image = cv2.imread(img_path)

#         # Draw the boxes and labels on the image
#         image_with_boxes = draw_boxes(image.copy(), result)

#         # Save the image with bounding boxes to the output folder
#         output_path = os.path.join(output_folder, f"ocr_output_{filename}")
#         cv2.imwrite(output_path, image_with_boxes)
#         print(f"Processed image saved at {output_path}")

# print("All images processed.")








#========================== for single image ==========================



import numpy as np
import cv2
import matplotlib.pyplot as plt
from paddleocr import PaddleOCR
from pprint import pprint

# Initialize PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False, verbose=False)  # Use English as an example

# Test a simple image
img_path = '/home/areebadnan/Areeb_code/work/Atheritia/Scripts/OCR_detection/input_images/testing_images/5f63aed26b04d6002afa6bf9.jpg'  # Ensure this image exists
result = ocr.ocr(img_path, rec=True)

# Print the result
print(f"The predicted text box of {img_path} are as follows.")

# List of logos
logos = ['betway', 'berlin', 'aok', ...]  # Ensure all logos are in lowercase

# Extract and print text, confidence score, and bounding box details
original_results = [[]]  # Initialize the list to store results in the same format as `result`

for line in result[0]:
    bbox, (text, confidence) = line
    # Convert detected text to lowercase and strip spaces
    processed_text = text.lower().strip()
    
    # Check if the processed text is in the list of logos
    if processed_text in logos:
        # Append the bbox, text, and confidence in the required format
        original_results[0].append((bbox, (text, confidence)))
        print(f"Detected Text: {text}, Confidence: {confidence:.2f}, Bounding Box: {bbox}")

# Load the image
image = cv2.imread(img_path)

# Function to draw bounding boxes and labels
def draw_boxes(image, result):
    for line in result[0]:  # Process each detected box and label
        bbox, (text, confidence) = line
        bbox = np.array(bbox, dtype=np.int32).reshape((-1, 1, 2))  # Convert bbox points to the correct format

        # Draw the bounding box on the image
        cv2.polylines(image, [bbox], isClosed=True, color=(0, 0, 255), thickness=3)

        # Calculate the position for the text
        text_position = (bbox[0][0][0], bbox[0][0][1] - 10)
        
        # Get the size of the text to create a rectangle background
        (text_width, text_height), _ = cv2.getTextSize(f"{text} ({confidence:.2f})", 
                                                         cv2.FONT_HERSHEY_SIMPLEX, 
                                                         0.7, 2)
        
        # Draw a blue rectangle behind the text
        rectangle_start = (text_position[0], text_position[1] - text_height - 10)
        rectangle_end = (text_position[0] + text_width, text_position[1])
        cv2.rectangle(image, rectangle_start, rectangle_end, (255, 0, 0), thickness=cv2.FILLED)

        # Place the label with confidence score above the bounding box
        cv2.putText(image, f"{text} ({confidence:.2f})", text_position,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    return image

# Draw the boxes and labels on the image
image_with_boxes = draw_boxes(image.copy(), original_results)

# Save the image with bounding boxes
output_path = '/home/areebadnan/Areeb_code/work/Atheritia/Scripts/OCR_detection/output_images/pic1.jpg'
cv2.imwrite(output_path, image_with_boxes)
print(f"Image saved at {output_path}")
