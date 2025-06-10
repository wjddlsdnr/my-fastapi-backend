import cv2
import numpy as np
import easyocr

reader = easyocr.Reader(['ko', 'en'])

def highlight_text_in_image(image_path, keyword, output_path):
    image = cv2.imread(image_path)
    results = reader.readtext(image)
    found = False
    for bbox, text, conf in results:
        if keyword.lower() in text.lower():
            top_left = tuple(map(int, bbox[0]))
            bottom_right = tuple(map(int, bbox[2]))
            cv2.rectangle(image, top_left, bottom_right, (0, 0, 255), 2)
            found = True
    if found:
        cv2.imwrite(output_path, image)
        return output_path
    else:
        return None
