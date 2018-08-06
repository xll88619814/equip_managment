import os
import cv2
import numpy as np
from skimage import measure
from detectnum import detect_tags

DEBUG_DIR = os.path.join(os.path.dirname(__file__), 'debug')
DEBUG = False

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


def detect_redlight(im, im_name):
    # Find red regions
    red_hue_low = [150, 178, 100]
    red_hue_high = [180, 255, 200]
    hsv_image = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    # Get lower hue
    kernel_size = (5,5)
    mask_lower= create_hue_mask(hsv_image, red_hue_low, red_hue_high, kernel_size)
    if DEBUG:
        result_image_path = os.path.join(DEBUG_DIR, im_name + "_redlight.jpg")
        cv2.imwrite(result_image_path, mask_lower)
    # Find suitable equipment labels
    labels = measure.label(mask_lower, connectivity=2)
    pro = measure.regionprops(labels)
    ok = True
    lights = []
    if len(pro) > 0:
        ok = False
        for p in pro:
            (x, y) = p.centroid
            radius = x - p.bbox[0] + 10
            lights.append((int(y), int(x), int(radius)))
    return ok, lights

def selectu(u_num, u_boxes):
    unum = []
    uboxes = []
    for ind, u in enumerate(u_num):
        if len(u) > 2:
            unum.append(u[:2])
            uboxes.append(u_boxes[ind])
        elif len(u) == 2:
            unum.append(u)
            uboxes.append(u_boxes[ind])
    clusters = []
    cluster = []
    cluster.append((unum[0], uboxes[0]))
    for ind, box in enumerate(uboxes[1:]):
        if abs(box[0] - uboxes[ind][0]) < 100:
            cluster.append((unum[ind+1], box))
        else:
            clusters.append(cluster)
            cluster = []
            cluster.append((unum[ind+1], box))
    clusters.append(cluster)
    for ind, c in enumerate(clusters):
        if len(c) == 2:
            if c[0][0] == c[1][0]:
                del clusters[ind][1]
            else:
                del clusters[ind]
    if len(clusters) >= 2:
        set_unum = [clusters[0][0][0], clusters[-1][0][0]]
        boxes = [clusters[0][0][1], clusters[-1][0][1]]
    else:
        set_unum = boxes =  []

    return set_unum, boxes

def findfirstpoint(im, uboxes, im_name, lower_hue_low, lower_hue_high, w_min, w_max, DEBUG):

    hsv_image = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

    kernel_size = (10,10)
    mask_lower= create_hue_mask(hsv_image, lower_hue_low, lower_hue_high, kernel_size)
    if DEBUG:
        result_image_path = os.path.join(DEBUG_DIR, im_name + "_equip.jpg")
        cv2.imwrite(result_image_path, mask_lower)
    labels = measure.label(mask_lower, connectivity=2)
    pro = measure.regionprops(labels)
    min_width = 1000
    for p in pro:
        (x1, y1, x2, y2) = p.bbox
        width = y2 - y1
        if w_max > width > w_min:
            if min_width > width:
                min_width = width
    boxes = []
    firstx = uboxes[0][2] - 10
    for p in pro:
        (x1, y1, x2, y2) = p.bbox
        if min_width <= y2-y1 <= min_width + 23:
            boxes.append(p.bbox)
    if len(boxes) == 0:
        return firstx
    else:
        dist = abs(boxes[0][0] - uboxes[0][2])
        for p in boxes[1:]:
            (x1, y1, x2, y2) = p
            if abs(x1 - uboxes[0][2]) < dist:
                dist = abs(x1 - uboxes[0][2])
                firstx = x1 - 5
    return firstx

def findregion(im, im_name, DEBUG):

    # Find u tags from up image
    lower_hue_low = [23, 127, 70]
    lower_hue_high = [31, 255, 230]
    hsv_image = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    kernel_size = (10, 10)
    mask_lower= create_hue_mask(hsv_image, lower_hue_low, lower_hue_high, kernel_size)
    if DEBUG:
        result_image_path = os.path.join(DEBUG_DIR, im_name + "_mask_jigui.jpg")
        cv2.imwrite(result_image_path, mask_lower)
    labels = measure.label(mask_lower, connectivity=2)
    pro = measure.regionprops(labels)

    # remain good regions and detect u num
    im_copy = im.copy()
    uboxes = []
    utags = []
    umasks = []
    im_height, im_width = im_copy.shape[0:2]
    i = 0
    for ind, p in enumerate(pro):
        (x1, y1, x2, y2) = p.bbox
        if 40 < y2-y1 < 60 and 36 < x2-x1 < 50 and y2<0.9*im_width:# and x1>0.02*im_height:#: and y1>0.1*im_width :
            i += 1
            if x1 >= 5:
                utags.append(im[x1-5:x2+5,y1-5:y2+5,:])
                umasks.append(mask_lower[x1-5:x2+5,y1-5:y2+5])
            else:
                utags.append(im[x1:x2+5,y1-5:y2+5,:])
                umasks.append(mask_lower[x1:x2+5,y1-5:y2+5])
            uboxes.append(p.bbox)
            cv2.rectangle(im_copy, (y1, x1), (y2, x2), (0, 0, 255), 3)
            print('u:', p.bbox)

    if DEBUG:
        result_image_path = os.path.join(DEBUG_DIR, im_name + ".jpg")
        cv2.imwrite(result_image_path, im_copy)
    #print('utags:',len(utags))

    detect = detect_tags(type_tag ='u',ratio=0.6,thresh_w=[15, 52],thresh_h=[45, 78],thresh_gap=55,DEBUG=DEBUG,DEBUG_DIR=DEBUG_DIR)
    u_num = detect.detect_num(utags, im_name, umasks)
    print('u_num:',u_num)
    if u_num == ['']:
        return False, [], [], [], []
    else:
        set_unum, uboxes = selectu(u_num, uboxes)
    print('set_unum:',set_unum)
    #print('uboxes:',uboxes)

    ok = True
    up_u = 0
    low_u = 0
    region = im.copy()
    #print('jigui tags is: ', len(uboxes))
    # adjust jigui redion
   
    if len(set_unum) == 2 and len(uboxes) == 2:
        #print('u tag is two....................')
        up_u = int(set_unum[0])
        low_u = int(set_unum[1])
       
        firstx = findfirstpoint(im, uboxes, im_name, lower_hue_low, lower_hue_high, 130, 170, DEBUG)

        #print('uboxes:', uboxes[0][2]-20, 'first:',firstx)
        region = im[firstx:uboxes[1][2],:,:]
        width = int(region.shape[1] * 800 / region.shape[0])
        region = cv2.resize(region,(width, 800),interpolation=cv2.INTER_CUBIC)
        u_point = firstx
        if DEBUG:
            result_image_path = os.path.join(DEBUG_DIR, im_name + "_jigui.jpg")
            cv2.imwrite(result_image_path, region)
    else:
        ok = False
        u_point = 0
        #print('detect u incorrectly')

    return ok, region, up_u, low_u, u_point

def findregion_below(im, im_name, DEBUG):
    # Find u tags in below image
    lower_hue_low = [23, 127, 70]
    lower_hue_high = [31, 255, 230]
    hsv_image = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    kernel_size = (10, 10)
    mask_lower= create_hue_mask(hsv_image, lower_hue_low, lower_hue_high, kernel_size)
    if DEBUG:
        result_image_path = os.path.join(DEBUG_DIR, im_name + "_mask_jigui.jpg")
        cv2.imwrite(result_image_path, mask_lower)
    labels = measure.label(mask_lower, connectivity=2)
    pro = measure.regionprops(labels)

    # remain good regions and detect u num
    im_copy = im.copy()
    uboxes = []
    utags = []
    umasks = []
    im_height, im_width = im_copy.shape[0:2]
    i = 0
    for ind, p in enumerate(pro):
        (x1, y1, x2, y2) = p.bbox
        if 40 < y2-y1 < 100 and 40 < x2-x1 < 75:# and y2<0.9*im_width and y1>0.1*im_width and x1>0.1*im_height:
            i += 1
            utags.append(im[x1-5:x2+5,y1-5:y2+5,:])
            uboxes.append(p.bbox)
            umasks.append(mask_lower[x1-5:x2+5,y1-5:y2+5])
            cv2.rectangle(im_copy, (y1, x1), (y2, x2), (0, 0, 255), 3)
            #print('u:', p.bbox)
    if DEBUG:
        result_image_path = os.path.join(DEBUG_DIR, im_name + ".jpg")
        cv2.imwrite(result_image_path, im_copy)
    #print('utags:',len(utags))

    detect = detect_tags(type_tag ='u',ratio=0.6,thresh_w =[15, 45],thresh_h=[46, 70],thresh_gap=55,DEBUG=DEBUG, DEBUG_DIR=DEBUG_DIR)
    u_num = detect.detect_num(utags, im_name, umasks)
    print('u_num:',u_num)
    if u_num == ['']:
        return False, [], [], [], []
    else:
        set_unum, uboxes = selectu(u_num, uboxes)
    print('set_unum:',set_unum)
    #print('uboxes:',uboxes)

    ok = True
    up_u = 0
    low_u = 0
    region = im.copy()
    #print('jigui tags is: ', len(uboxes))
    # adjust jigui redion
    
    if len(set_unum) == 2 and len(uboxes) == 2:
        up_u = int(set_unum[0])
        low_u = int(set_unum[1])

        firstx = findfirstpoint(im, uboxes, im_name, lower_hue_low, lower_hue_high, 180, 270, DEBUG)

        #print('uboxes:', uboxes[0][2]-20, 'first:',firstx)
        region = im[firstx:uboxes[1][2],:,:]
        width = int(region.shape[1] * 800 / region.shape[0])
        region = cv2.resize(region,(width, 800),interpolation=cv2.INTER_CUBIC)
        u_point = firstx
        if DEBUG:
            result_image_path = os.path.join(DEBUG_DIR, im_name + "_jigui.jpg")
            cv2.imwrite(result_image_path, region)
    else:
        ok = False
        u_point = 0
        #print('detect u incorrectly')

    return ok, region, up_u, low_u, u_point



def isswitch(im, im_name, DEBUG):
    lower_hue_low = [23, 127, 70]
    lower_hue_high = [31, 255, 230]
    hsv_image = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    kernel_size = (10, 10)
    mask_lower= create_hue_mask(hsv_image, lower_hue_low, lower_hue_high, kernel_size)
    if DEBUG:
        result_image_path = os.path.join(DEBUG_DIR, im_name + "_switch.jpg")
        cv2.imwrite(result_image_path, mask_lower)
    labels = measure.label(mask_lower, connectivity=2)
    pro = measure.regionprops(labels)
    switchboxes = []
    switchtags = []
    switchmasks = []
    #i = 0
    for p in pro:
        (x1, y1, x2, y2) = p.bbox
        if 300 > (y2-y1) > 240 and 45 > (x2-x1) > 25:
            #print('switch:',p.bbox)
            #i += 1
            switchboxes.append(p.bbox)
            switchtags.append(im[x1-5:x2+5,y1-5:y2+5,:])
            switchmasks.append(mask_lower[x1-5:x2+5,y1-5:y2+5])
            #cv2.imshow('switch', im[x1-5:x2+5,y1-5:y2+5,:])
            #cv2.waitKey(0)
    switch = False
    if len(switchboxes) > 0:
        #print('switchboxes:', switchboxes)
        switch = True

    return switch, switchboxes, switchtags, switchmasks


def detecting(im_url, image_type, debug=None):
    im = cv2.imread(im_url)
    #image_type = im_url.split('/')[-1].split('_')[1].split('.')[0]
    im_name = im_url.split('/')[-1].split('.')[0]
    image_file = os.path.join('/opt/gxxj_robot/temp', im_name + '.jpg')

    DEBUG = debug
    image = im.copy()
    final_result = []

    # judge if it is switch
    switch, switchboxes, switchtags, switchmasks = isswitch(im, im_name, DEBUG)
    if image_type == '0':
        ok, img, up_u, low_u, u_point = findregion(im, im_name, DEBUG)
        print('switch: ', switch)
        if switch and ok == False:
            print('switch.....................................')
            ok = True
            detect = detect_tags(type_tag ='switch',ratio=0.6,thresh_w=[25, 65],thresh_h=[40, 65],thresh_gap=75,DEBUG=DEBUG, DEBUG_DIR=DEBUG_DIR)
            result = detect.detect_num(switchtags, im_name, switchmasks)
            print(result)
            # visulize
            if result:
                for ind, res in enumerate(result):
                    if len(res) >= 2:
                        cv2.putText(im, 'ID: '+res[0]+' U: '+res[1], (switchboxes[ind][1], u_point+switchboxes[ind][0]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        final_result.append({'ID':res[0], 'U':res[1]})
                image_file = os.path.join('code/result', im_name + '.jpg')
                cv2.imwrite(image_file, im)
        elif switch == False and ok ==True :
                print('start detect IP..................')
                sum_u = up_u - low_u
                height_u = img.shape[0]*1.0/sum_u
                #print('sum_u:', sum_u, 'height_u: ',height_u)

                # Find yellow regions
                lower_hue_low = [25, 127, 70]
                lower_hue_high = [31, 255, 230]
                hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

                kernel_size = (5, 5)
                mask_lower= create_hue_mask(hsv_image, lower_hue_low, lower_hue_high, kernel_size)
                if DEBUG:
                    result_image_path = os.path.join(DEBUG_DIR, im_name + "_equipment.jpg")
                    cv2.imwrite(result_image_path, mask_lower)

                labels = measure.label(mask_lower, connectivity=2)
                pro = measure.regionprops(labels)
                # remain good yellow regions
                imm = img.copy()
                min_width = 1000
                for p in pro:
                    (x1, y1, x2, y2) = p.bbox
                    width = y2 - y1
                    if 170 > width > 130 and 40>x2-x1>20:
                        if min_width > width:
                            min_width = width
                print('min_width:', min_width)
                box = []
                im_copy = img.copy()
                for p in pro:
                    (x1, y1, x2, y2) = p.bbox
                    
                    if min_width <= y2-y1 <= min_width + 20:
                        box.append(p.bbox)
                        #cv2.rectangle(im_copy, (y1, x1), (y2, x2), (0, 0, 255), 3)
                        #print(p.bbox)
                        #subim = img[x1:x2, y1:y2, :]
                        #cv2.imshow('tags', subim)
                        #cv2.waitKey(0)
                #cv2.imwrite(im_name+'.jpg', im_copy)
                if len(box) > 0:
                    iptags = []
                    ipboxes = []
                    ipmasks = []
                    u = []
                    for i, b in enumerate(box):
                        if i%2 == 0:
                     	    y1 = 0 if b[0]-5 <0 else b[0]-5
                            y2 = img.shape[0] if b[2]+5 >img.shape[0] else b[2] + 5
                            subim = img[y1:y2, b[1]-5:b[3]+5, :]
                            submask = mask_lower[y1:y2, b[1]-5:b[3]+5]
                            if DEBUG:
                                result_image_path = os.path.join(DEBUG_DIR, im_name+'_'+str(i)+'_'+'tag.jpg')
                                cv2.imwrite(result_image_path, submask)
                            iptags.append(subim)
                            ipmasks.append(submask)
                            ipboxes.append(b)

                            # computer u
                            end_u = int(round(up_u - b[0]/height_u) -1)
                            start_u = int(round(up_u - box[i+1][2]/height_u))
                            u.append((start_u, end_u))
                            #print(end_u, start_u)

                    detect = detect_tags(type_tag='ip',ratio=0.6,thresh_w=[15, 40],thresh_h=[40, 62],thresh_gap=47,DEBUG=DEBUG,DEBUG_DIR=DEBUG_DIR)
                    result = detect.detect_num(iptags, im_name, ipmasks)
                    
                    if result:
                        for ind, res in enumerate(result):
                            u_list = np.arange(u[ind][0], u[ind][1]+1)
                            cv2.putText(im, 'ID: {}, U: {}'.format(res, u_list), (ipboxes[ind][1], u_point+ipboxes[ind][0]+80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                            final_result.append({'ID':res, 'U':u[ind]})  
                        #cv2.imwrite(image_file, im)
        #else:
        #    cv2.imwrite(image_file, im)

    else:
        # the image taken by the camera below
        # get jigui region and max u and min u
        print('low...........................................................')
        ok, img, up_u, low_u, u_point = findregion_below(im, im_name, DEBUG)
        if ok:
            print('start detect IP..................')
            sum_u = up_u - low_u
            height_u = img.shape[0]*1.0/sum_u
            #print('sum_u:', sum_u, 'height_u: ',height_u)

            # Find IP regions
            lower_hue_low = [23, 127, 50]
            lower_hue_high = [31, 255, 255]
            hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            kernel_size = (10,10)
            mask_lower= create_hue_mask(hsv_image, lower_hue_low, lower_hue_high, kernel_size)
            if DEBUG:
                result_image_path = os.path.join(DEBUG_DIR, im_name + "_equipment.jpg")
                cv2.imwrite(result_image_path, mask_lower)

            labels = measure.label(mask_lower, connectivity=2)
            pro = measure.regionprops(labels)
            # remain good yellow regions
            imm = img.copy()
            min_width = 1000
            for p in pro:
                (x1, y1, x2, y2) = p.bbox
                width = y2 - y1
                if 270 > width > 180:
                    if min_width > width:
                        min_width = width
            print('min_width:', min_width)
            box = []
            im_copy = img.copy()
            for p in pro:
                (x1, y1, x2, y2) = p.bbox         
                if min_width <= y2-y1 <= min_width + 23:
                    box.append(p.bbox)
                    #cv2.rectangle(im_copy, (y1, x1), (y2, x2), (0, 0, 255), 3)
                    #print(p.bbox)
                    #subim = img[x1:x2, y1:y2, :]
                    #cv2.imshow('tags', subim)
                    #cv2.waitKey(0)
            #cv2.imwrite(im_name+'.jpg', im_copy)
            if len(box) > 0:
                iptags = []
                ipboxes = []
                ipmasks = []
                u = []
                for i, b in enumerate(box):
                    if i%2 == 0:
                     	y1 = 0 if b[0]-5 <0 else b[0]-5
                        y2 = img.shape[0] if b[2]+5 >img.shape[0] else b[2] + 5
                        subim = img[y1:y2, b[1]-5:b[3]+5, :]
                        submask = mask_lower[y1:y2, b[1]-5:b[3]+5]
                        if DEBUG:
                            result_image_path = os.path.join(DEBUG_DIR, im_name+'_'+str(i)+'_'+'tag.jpg')
                            cv2.imwrite(result_image_path, submask)
                        iptags.append(subim)
                        ipmasks.append(submask)
                        ipboxes.append(b)

                        end_u = int(round(up_u - b[0]/height_u) -1)
                        start_u = int(round(up_u - box[i+1][2]/height_u))
                        u.append((start_u, end_u))
                detect = detect_tags(type_tag = 'ip',ratio=0.6,thresh_w=[14, 44],thresh_h=[42, 60],thresh_gap=48,DEBUG=DEBUG, DEBUG_DIR=DEBUG_DIR)
                result = detect.detect_num(iptags, im_name, ipmasks)
                print('u_point',u_point)
                if result:
                    for ind, res in enumerate(result):
                        u_list = np.arange(u[ind][0], u[ind][1]+1)
                        cv2.putText(im, 'ID: {}, U: {}'.format(res, u_list), (ipboxes[ind][1], u_point+ipboxes[ind][0]+80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        final_result.append({'ID':res, 'U':u[ind]})
                    #image_file = os.path.join('code/result', im_name + '.jpg')
                    #cv2.imwrite(image_file, im)
    #print('final result: ', final_result)
    cv2.imwrite(image_file, im)
    return ok, final_result, image_file


