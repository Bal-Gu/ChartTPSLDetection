import re
from colorthief import ColorThief
import cv2
import pytesseract
import numpy as np
from collections import Counter


def extract_text_regions(image):
    # Calculate the width of the image
    height, width, _ = image.shape

    # Define the region of interest (ROI) for the right side of the image (1/8th of the width)
    right_side = image[:, width - int(width / 8):]
    # Apply inverse binarization (thresholding) to the right_side image

    gray_right_side = cv2.cvtColor(right_side, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(gray_right_side, 100, 200)
    cv2.imwrite('gray_right_side2.jpg', canny)
    text_data = pytesseract.image_to_data(canny, output_type=pytesseract.Output.DICT, config="--psm 6 digits")

    return text_data


def detect_background_color(image):
    cv2.imwrite('hsv.jpg', image)
    color_thief = ColorThief("hsv.jpg")
    # get the dominant color
    return color_thief.get_color(quality=1)


def post_process_text_data(text_data):
    # Initialize an empty list to store the filtered and corrected text entries
    filtered_text_data = []

    # Initialize variables to keep track of the previous valid number and its index
    prev_number = None
    prev_index = -1

    # Iterate through the text data entries
    for i, text_entry in enumerate(text_data['text']):
        text = text_entry.strip()

        # Ignore empty text entries
        if not text:
            filtered_text_data.append(0)
            continue
        if str(text).count(".") > 1:
            split = str(text).split(".")
            text = split[0] + "." + split[1]
        # Check if the text is a valid number (handles both integer and floating-point numbers)
        if str(text).split(".")[0] == "":
            if i == 0:
                next = i + 1
            else:
                next = i - 1
            text = (text_data["text"][next]).split(".")[0] + "." + str(text).split(".")[1]
        try:
            number = float(text)
        except ValueError:
            # Ignore non-numeric text entries
            filtered_text_data.append(0)
            continue

        # Check if the number is greater than the previous one (if available)
        counter = 0
        while prev_number is not None and number >= prev_number:
            # Correct the number based on the previous one
            number_str = list(str(number))
            prev_str = list(str(prev_number))
            number_str[counter] = prev_str[counter]
            number = float("".join(number_str))
            counter += 1
        # Append the corrected number and its associated text position data
        filtered_text_data.append(number)

        # Update the previous number and index
        prev_number = number

    return filtered_text_data


def group_by_color(text_color):
    group = []
    for t, c in text_color:
        for element in group:
            for c_2 in element.keys():
                print("{}\t{}".format(c_2,c))
                if c_2[0] - 10 <= c[0] <= c_2[0] + 10 and c_2[1] - 10 <= c[1] <= c_2[
                    1] + 10 and c_2[2] - 10 <= c[2] <= c_2[2] + 10:
                    element[c_2].append(t)
                    continue
        group.append({c: [t]})
    return group

def get_text_region_colors(image, text_data):
    colors = []

    # Group characters with similar top and bottom positions
    group = []
    cleaned_text = post_process_text_data(text_data)
    for i in range(len(text_data['text'])):
        if cleaned_text[i] == 0:
            continue
        min_top = text_data['top'][i]
        max_bottom = int(text_data['top'][i] + text_data['height'][i])
        x1 = int(text_data["left"][i])
        x2 = int(text_data["width"][i]) + x1
        char = cleaned_text[i]
        y1 = max(min_top - 5, 0)
        y2 = min(max_bottom + 5, image.shape[0])
        text_region = image[y1:y2, len(image[1]) - x2:len(image[1]) - x1]

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

print(group_by_color(text_colors))

print("Detected background colors for text regions on the right side:")
for char, color in text_colors:
    print(f"Text: {char}, Background Color: {color}")
