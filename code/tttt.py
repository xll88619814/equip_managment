import cv2
import numpy as np
from math import *
from skimage import measure

def create_hue_mask(image, lower_color, upper_color, kernel_size):
    lower = np.array(lower_color, np.uint8)
    upper = np.array(upper_color, np.uint8)
    # Create a mask from the colors
    mask = cv2.inRange(image, lower, upper)
    # open and close
    if kernel_size:
        kernel=cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        dilated = cv2.dilate(mask, kernel)
        eroded = cv2.erode(dilated, kernel) 
        return eroded
    else:
        return mask


img = cv2.imread(image_path)

lower_hue_low = [25, 127, 80]
lower_hue_high = [31, 255, 230]
hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

kernel_size = (10,10)
mask_lower= create_hue_mask(hsv_image, lower_hue_low, lower_hue_high, kernel_size)

labels = measure.label(mask_lower, connectivity=2)
pro = measure.regionprops(labels)
            
box = []
for p in pro:
    (x1, y1, x2, y2) = p.bbox
    if y2-y1 > 100:
        box.append(x1)
        box.append(x2)
x = mean(box)
    
