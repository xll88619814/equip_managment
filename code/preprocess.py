import cv2, os
import numpy as np
from math import *
from PIL import ImageEnhance, Image

if __name__ =='__main__':
    test_dir = 'fish/'
    for imagename in os.listdir(test_dir):
        print('image name is : ', imagename)
        image = test_dir + imagename
        im =cv2.imread(image)
        im = Image.fromarray(im)

        # enhance sharpness
        #enh_sha = ImageEnhance.Color(im)
        #sharpness = 3.0
        #im = enh_sha.enhance(sharpness)
        

        enh_con = ImageEnhance.Contrast(im)  
        contrast = 2.5  
        img_contrasted = enh_con.enhance(contrast)  
        im = np.array(img_contrasted)

        cv2.imwrite(imagename, im)
