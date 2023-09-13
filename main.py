import glob
import re

import cv2
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


jpg_files = glob.glob("*.jpg")
for jpg_file in jpg_files:
    print("=================================")
    getsymbol(cv2.imread(jpg_file))
    print("=================================")

