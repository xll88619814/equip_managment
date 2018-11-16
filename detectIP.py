import os
import cv2
import sys
import json
sys.path.append('../')
sys.path.append('code/train_nummodel')
from code.locate_u import detecting
from code.locate_u import detect_redlight

if __name__ =='__main__':
    test_dir = 'code/test_images/'
    for imagename in os.listdir(test_dir):
        print('image name is : ', imagename)
        image = test_dir + imagename   
        #image_type = image.split('/')[-1].split('_')[1].split('.')[0]
        ok, result, result_file, u_range, light_ok, light_u = detecting(image, True)
        final_result = json.dumps(result)
        print('final result: ', final_result, u_range, ok)


'''
if __name__ =='__main__':
    test_dir = 'code/test_images/'
    for imagename in os.listdir(test_dir):
        im = cv2.imread(test_dir + imagename)
        height, width, chanel = im.shape
        image = cv2.resize(im,(int(width*2/1.6), int(height*2/1.6)),interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(imagename, image)
'''
