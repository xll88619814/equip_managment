#coding: utf-8
import os
import sys
import json
import numpy as np
sys.path.append('../')
sys.path.append('code/train_nummodel')
from code.locate_u import detecting


if __name__ =='__main__':
    #map1 = np.load("code/map1.npy")
    #map2 = np.load("code/map2.npy")
    #print(map2.shape)
    test_dir = 'code/test_images/'
    angle = '0'
    for imagename in os.listdir(test_dir):
        print('image name is : ', imagename)
        image = test_dir + imagename   
        #image_type = image.split('/')[-1].split('_')[1].split('.')[0]
        ok, result, result_file, u_range, light_ok, light_u = detecting(image, angle, True)
        final_result = json.dumps(result)
        print('final result: ', final_result, u_range, ok)

