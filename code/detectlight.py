import os
import cv2
import numpy as np
from skimage import measure

def create_hue_mask(image, lower_color, upper_color, kernel_size):
    lower = np.array(lower_color, np.uint8)
    upper = np.array(upper_color, np.uint8)
    # Create a mask from the colors
    mask = cv2.inRange(image, lower, upper)
    # open and close
    if kernel_size:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
        dilated = cv2.dilate(mask, kernel)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 13))
        eroded = cv2.erode(dilated, kernel)
        return eroded
    else:
        return mask

def detect_redlight(im, im_name, DEBUG, DEBUG_DIR):
    # Find red regions
    red_hue_low = [150, 60, 80]
    red_hue_high = [180, 255, 255]
    height, width = im.shape[:2]
    hsv_image = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    # Get lower hue
    kernel_size = []
    mask_lower1= create_hue_mask(hsv_image, red_hue_low, red_hue_high, kernel_size)

    red_hue_low = [0, 100, 80]
    red_hue_high = [13, 255, 230]
    height, width = im.shape[:2]
    hsv_image = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    # Get lower hue
    kernel_size = []
    mask_lower2= create_hue_mask(hsv_image, red_hue_low, red_hue_high, kernel_size)  
    mask_lower = mask_lower1 + mask_lower2
    if DEBUG:
        result_image_path = os.path.join(DEBUG_DIR, im_name + "_redlight.jpg")
        cv2.imwrite(result_image_path, mask_lower)
    # Find suitable equipment labels
    labels = measure.label(mask_lower, connectivity=2)
    pro = measure.regionprops(labels)
    ok = True
    lights = []
    if len(pro) > 0:
        for p in pro:
            (x, y) = p.centroid
            w = p.bbox[3]-p.bbox[1]
            h = p.bbox[2]-p.bbox[0]
            ratio = max(w, h) / min(w, h)
            if 130 >= p.area >= 40 and ratio <= 3 and 0.1*width <= y <= 0.85*width :
                ok = False
                radius = x - p.bbox[0] + 10
                lights.append((int(y), int(x), int(radius), int(p.area)))
                print('light', p.area, ratio)
                
    return ok, lights


def detect_light(im, im_name, up_u, low_u_new, low_point, up_point, DEBUG, DEBUG_DIR):
    light_ok = True
    light_u = []
    light_ok, light_locat = detect_redlight(im, im_name, DEBUG, DEBUG_DIR)
    sum_u = up_u - low_u_new
    height_u = (low_point - up_point) * 1.0 / sum_u
    if not light_ok:
        for light in light_locat:
            if up_point <= light[1] <= low_point:
                print('light: ', light[3])
                curr_u = int(up_u - np.ceil((light[1] - up_point) / height_u))
                light_u.append(curr_u)
                cv2.circle(im, (light[0], light[1]), light[2], (0,0,255), 5)
                cv2.putText(im, 'U: {}'.format(curr_u), (light[0], light[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return im, light_ok, light_u
