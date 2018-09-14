import cv2, os
import numpy as np
from math import *
from PIL import ImageEnhance, Image

if __name__ =='__main__':
    test_dir = 'test_images/'
    for imagename in os.listdir(test_dir):
        print('image name is : ', imagename)
        image = test_dir + imagename
        im =cv2.imread(image)
        im = Image.fromarray(im)

        # enhance sharpness
        enh_sha = ImageEnhance.Color(im)
        sharpness = 3.0
        im = enh_sha.enhance(sharpness)
        im = np.array(im)

        cv2.imwrite(imagename, im)