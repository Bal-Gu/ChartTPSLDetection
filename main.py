import glob
import re
from collections import Counter

import cv2
import numpy as np
import pytesseract



def getsymbol(image):
    # Extract text in the upper left corner
    height, width, _ = image.shape
    upper_left_corner_region = image[0:int(height/4), 0:int(width/2)]
    upper_left_text = pytesseract.image_to_string(upper_left_corner_region)
    symbols = re.findall(r"(BINANCE\s*([A-Z]{3,}))|(\b([A-Z]+)\s*/\s*([A-Z]+))|(([A-Z]+)USDT)",upper_left_text)[0]
    for s in symbols:
        if len(s) >= 3 and str(s).isalpha():
            print(s)
            return str(s).split("USDT")[0]

# Define function to detect text with red or blue backgrounds
def detect_colored_background_text(image, color_lower, color_upper):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_image, color_lower, color_upper)
    result = cv2.bitwise_and(image, image, mask=mask)
    colored_background_text = pytesseract.image_to_string(result)
    return colored_background_text


def extract_text_regions(image):
    # Calculate the width of the image
    height, width, _ = image.shape

    # Define the region of interest (ROI) for the right side of the image
    right_side = image[:, width  - int(width/8):]

    # Convert the image to grayscale
    gray = cv2.cvtColor(right_side, cv2.COLOR_BGR2GRAY)

    # Perform OCR to extract text regions (numbers)
    text_regions = pytesseract.image_to_boxes(gray)

    return text_regions

def get_text_region_colors(image, text_regions):
    colors = []

    for region in text_regions.splitlines():
        region = region.split()
        if len(region) == 12:
            x, y, _, h = map(int, region[1:5])
            text = region[-1]

            # Extract the region of interest (ROI) containing the text
            text_region = image[y:y+h, :]

            # Calculate the mean color of the ROI
            mean_color = tuple(np.mean(text_region, axis=(0, 1)).astype(int))
            colors.append(mean_color)

    return colors


# Define lower and upper bounds for red and blue colors in HSV
red_lower = (0, 100, 100)
red_upper = (10, 255, 255)
blue_lower = (100, 100, 100)
blue_upper = (130, 255, 255)

# Detect text with red background
#red_background_text = detect_colored_background_text(image, red_lower, red_upper)

# Detect text with blue background
#blue_background_text = detect_colored_background_text(image, blue_lower, blue_upper)

# Example usage:
image = cv2.imread('photo_2023-09-05_13-08-12.jpg')

# Extract text regions (numbers) from the right side of the image
text_regions = extract_text_regions(image)

print(text_regions)
# Get the background colors for each text region
text_colors = get_text_region_colors(image, text_regions)

print("Background colors for text regions on the right side:")
for idx, color in enumerate(text_colors, 1):
    print(f"Text Region {idx}: {color}")

jpg_files = glob.glob("*.jpg")
for jpg_file in jpg_files:
    break

