import cv2
import pytesseract
import numpy as np
from collections import Counter

def extract_text_regions(image):
    # Calculate the width of the image
    height, width, _ = image.shape

    # Define the region of interest (ROI) for the right side of the image (1/8th of the width)
    right_side = image[:, width - int(width/8):]
    # Apply inverse binarization (thresholding) to the right_side image
    gray_right_side = cv2.cvtColor(right_side, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('gray_right_side.jpg', gray_right_side)
    image = cv2.threshold(gray_right_side, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]


    # Save the result (grayscale image with black and white text) to a file
    cv2.imwrite('black_white_text_mask.jpg', image)
    # Perform OCR to extract text regions (numbers)
    text_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT,config="--psm 6 digits")

    n_gray_right_side = cv2.cvtColor(right_side, cv2.COLOR_BGR2GRAY)
    img = cv2.bitwise_not(n_gray_right_side)
    cv2.imwrite('gray_right_side.jpg', img)
    n_image = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    n_text_data = pytesseract.image_to_data(n_image, output_type=pytesseract.Output.DICT, config="--psm 6 digits")
    for i in text_data:
        for data in n_text_data[str(i)]:
            text_data[str(i)].append(data)
    return text_data

def detect_background_color(image):
    # Define the color ranges for black, white, green, blue, yellow, and red
    color_ranges = {
        "black": ((0, 0, 0), (60, 60, 60)),
        "white": ((200, 200, 200), (255, 255, 255)),
        "green": ((0, 150, 0), (50, 255, 50)),
        "blue": ((0, 0, 150), (50, 50, 255)),
        "yellow": ((150, 150, 0), (255, 255, 50)),
        "red": ((150, 0, 0), (255, 50, 50))
    }

    # Convert the image to HSV color space for easier color detection
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Detect the color based on the defined ranges
    detected_color = None
    for color, (lower, upper) in color_ranges.items():
        lower = np.array(lower, dtype=np.uint8)
        upper = np.array(upper, dtype=np.uint8)
        mask = cv2.inRange(hsv_image, lower, upper)
        if mask.any():
            detected_color = color
            break

    return detected_color

def get_text_region_colors(image, text_data):
    colors = []

    # Group characters with similar top and bottom positions
    group = []
    for i in range(len(text_data['text'])):
        top, bottom = int(text_data['top'][i]), int(text_data['top'][i] + text_data['height'][i])
        if len(group) == 0 or (top - group[0][0] <= 2 and group[0][1] - bottom <= 2):
            group.append((top, bottom, text_data['text'][i]))
        else:
            if group:
                min_top = min(group, key=lambda x: x[0])[0]
                max_bottom = max(group, key=lambda x: x[1])[1]
                char = ''.join([item[2] for item in group])
                group = []
                y1 = max(min_top - 5, 0)
                y2 = min(max_bottom + 5, image.shape[0])
                text_region = image[y1:y2, :]

                # Detect the background color for the text region
                detected_color = detect_background_color(text_region)
                colors.append((char, detected_color))

    return colors

# Example usage:
image = cv2.imread('photo_2023-09-07_00-44-54.jpg')

# Extract text regions (numbers) and associated numbers from the right side of the image
text_data = extract_text_regions(image)

# Get the detected background colors for each text region
text_colors = get_text_region_colors(image, text_data)

print("Detected background colors for text regions on the right side:")
for char, color in text_colors:
    print(f"Text: {char}, Background Color: {color}")
