import os
import cv2
import sys
sys.path.append('train_nummodel')
from gettags import detecting

if __name__ =='__main__':
    test_dir = 'test_images/'
    for imagename in os.listdir(test_dir):
        print('image name is : ', imagename)
        image = test_dir + imagename   
        ok, result = detecting(image)
