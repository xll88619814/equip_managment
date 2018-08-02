import os, cv2
from PIL import ImageEnhance
from PIL import Image
import numpy as np
from skimage import measure
from ImagePreprocessing import pre_proc

'''
im1 = cv2.imread('2244.jpg')
hsv_image = cv2.cvtColor(im1, cv2.COLOR_BGR2HSV)
s = 10-np.mean(hsv_image[:,:,1])/10
print('s:', s)
image = Image.fromarray(im1)
bright =ImageEnhance.Brightness(image).enhance(0.5)
color =ImageEnhance.Color(image).enhance(int(s))
const =ImageEnhance.Contrast(color).enhance(5)
im_enh = np.asarray(color)
cv2.imwrite('result1.jpg', im_enh)

im2 = cv2.imread('24h.jpg')
hsv_image = cv2.cvtColor(im2, cv2.COLOR_BGR2HSV)
s = 10-np.mean(hsv_image[:,:,1])/10
print('s:', s)
image = Image.fromarray(im2)
bright =ImageEnhance.Brightness(image).enhance(0.5)
color =ImageEnhance.Color(image).enhance(int(s))
const =ImageEnhance.Contrast(color).enhance(5)
im_enh = np.asarray(color)
cv2.imwrite('result2.jpg', im_enh)
'''


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


if __name__ == '__main__':
    debug_dir = 'debuggg'
    lower_hue_low = [28, 102, 200]
    lower_hue_high = [33, 255, 255]
    im1 = cv2.imread('result1.jpg')
    hsv_image = cv2.cvtColor(im1, cv2.COLOR_BGR2HSV)
    kernel_size = (9, 9)
    mask_lower= create_hue_mask(hsv_image, lower_hue_low, lower_hue_high, kernel_size)
    if debug_dir:
        result_image_path = os.path.join(debug_dir, "mask_jigui1.jpg")
        cv2.imwrite(result_image_path, mask_lower)
    labels = measure.label(mask_lower, connectivity=2)
    pro = measure.regionprops(labels)
    i = 0
    im_name = '1'
    for p in pro:
        (x1,y1,x2,y2) = p.bbox
        if y2-y1>100 and x2-x1<50:
            i += 1
            subim = im1[x1-5:x2+5,y1-5:y2+5,:]
            submask = mask_lower[x1-5:x2+5,y1-5:y2+5]
            width = submask.shape[1] * 100 / submask.shape[0]
            subim = cv2.resize(subim,(width, 100),interpolation=cv2.INTER_CUBIC)
            submask = cv2.resize(submask,(width, 100),interpolation=cv2.INTER_CUBIC)
            cv2.imwrite('debuggg/'+im_name+'_'+str(i)+'_'+'ip.jpg', subim)
            cv2.imwrite('debuggg/'+im_name+'_'+str(i)+'_'+'ip_mask.jpg', submask)
            blurred = proc(subim)
            cv2.imwrite('debuggg/'+im_name+'_'+str(i)+'_'+'blurr.jpg', blurred)


    print('.................................')
    im2 = cv2.imread('result2.jpg')
    hsv_image = cv2.cvtColor(im2, cv2.COLOR_BGR2HSV)
    kernel_size = (9, 9)
    mask_lower= create_hue_mask(hsv_image, lower_hue_low, lower_hue_high, kernel_size)
    if debug_dir:
        result_image_path = os.path.join(debug_dir, "mask_jigui2.jpg")
        cv2.imwrite(result_image_path, mask_lower)
    labels = measure.label(mask_lower, connectivity=2)
    pro = measure.regionprops(labels)
    i = 0
    im_name = '2'
    for p in pro:
        (x1,y1,x2,y2) = p.bbox
        if y2-y1>100 and x2-x1<70:
            i += 1
            subim = im2[x1-5:x2+5,y1-5:y2+5,:]
            submask = mask_lower[x1-5:x2+5,y1-5:y2+5]
            width = submask.shape[1] * 100 / submask.shape[0]
            subim = cv2.resize(subim,(width, 100),interpolation=cv2.INTER_CUBIC)
            submask = cv2.resize(submask,(width, 100),interpolation=cv2.INTER_CUBIC)
            cv2.imwrite('debuggg/'+im_name+'_'+str(i)+'_'+'ip.jpg', subim)
            cv2.imwrite('debuggg/'+im_name+'_'+str(i)+'_'+'ip_mask.jpg', submask)
            blurred = proc(subim)
            cv2.imwrite('debuggg/'+im_name+'_'+str(i)+'_'+'blurr.jpg', blurred)
