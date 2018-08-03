import os
import cv2
import sys
sys.path.append('code/train_nummodel')
from code.gettags import detecting


if __name__ =='__main__':
    test_dir = 'code/test_images/'
    for imagename in os.listdir(test_dir):
        print('image name is : ', imagename)
        image = test_dir + imagename   
        ok, result = detecting(image, True)

'''
from code.gettags import selectu
if __name__ =='__main__':
    u_num = ['33','23','22', '11', '11']
    uboxes = [(1,1,100,100), (9,300, 100,400), (200,300,300,400), (391,1,500,100), (400,300,500,400)]
    set_unum, uboxes = selectu(u_num, uboxes)
    print('set_unum: ',set_unum)
    print('uboxes: ', uboxes)
'''
