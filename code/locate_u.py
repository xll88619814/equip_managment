import os
import cv2
import time
import numpy as np
from skimage import measure
from fisheye import undistort
from detectnum import detect_tags
from config import config

DEBUG_DIR = os.path.join(os.path.dirname(__file__), 'debug')
DEBUG = False

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

def detect_redlight(im, im_name, DEBUG):
    # Find red regions
    red_hue_low = [150, 80, 50]
    red_hue_high = [180, 255, 230]
    height, width = im.shape[:2]
    hsv_image = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    # Get lower hue
    kernel_size = []
    mask_lower1= create_hue_mask(hsv_image, red_hue_low, red_hue_high, kernel_size)

    red_hue_low = [0, 80, 50]
    red_hue_high = [15, 255, 230]
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
            if 150 >= p.area >= 30 and 0.1*width <= y <= 0.85*width:
                ok = False
                radius = x - p.bbox[0] + 10
                lights.append((int(y), int(x), int(radius), int(p.area)))
                #print('light', p.area)
                
    return ok, lights

def selectu(u_num, u_boxes):
    unum = []
    uboxes = []
    for ind, u in enumerate(u_num):
        if len(u) == 2:
            if int(u) <= 42 and int(u) >= 5:
                if len(u) > 2:
                    unum.append(u[:2])
                    uboxes.append(u_boxes[ind])
                elif len(u) == 2:
                    unum.append(u)
                    uboxes.append(u_boxes[ind])

    clusters = []
    cluster = []
    if len(uboxes) > 0:
        cluster.append([unum[0], uboxes[0]])
        for ind, box in enumerate(uboxes[1:]):
            if abs(box[0] - uboxes[ind][0]) < 100:
                cluster.append([unum[ind+1], box])
            else:
                clusters.append(cluster)
                cluster = []
                cluster.append([unum[ind+1], box])
        
        clusters.append(cluster)
        for ind, c in enumerate(clusters):
            if len(c) == 2:
                if c[0][0] == c[1][0]:
                    newx1 = (clusters[ind][0][1][0]+clusters[ind][1][1][0])/2
                    newx2 = (clusters[ind][0][1][2]+clusters[ind][1][1][2])/2
                    y1 = clusters[ind][0][1][1]
                    y2 = clusters[ind][0][1][3]
                    clusters[ind][0][1] = (newx1, y1, newx2, y2)
                    del clusters[ind][1]
                    
                else:
                    del clusters[ind]
    if len(clusters) >= 2:
        set_unum = [clusters[0][0][0], clusters[-1][0][0]]
        boxes = [clusters[0][0][1], clusters[-1][0][1]]
    else:
        set_unum = boxes = []

    return set_unum, boxes

def findminbox(im, x1, x2):
    # cv2.imshow('box', im)
    # cv2.waitKey(0)
    w = im.shape[1]
    ratio = np.sum(im == 255)*1.0/((x2-x1)*w)
    # print(ratio)
    while ratio < 0.7:
        if x2-x1 < 20:
            return 0, 0
        ratio1 = np.sum(im[2:, :] == 255) * 1.0 / ((x2-x1-1) * w)
        ratio2 = np.sum(im[:-2, :] == 255) * 1.0 / ((x2-x1-1) * w)
        if ratio1 > ratio2:
            im = im[2:, :]
            ratio = ratio1
            x1 += 2
        elif ratio1 < ratio2:
            im = im[:-2, :]
            ratio = ratio2
            x2 -= 2
        else:
            im = im[2:-2, :]
            ratio = ratio1
            x1 += 2
            x2 -= 2
        print(ratio)

    #cv2.imshow('box', im)
    #cv2.waitKey(0)
    # print(x1,x2, im.shape)
    return x1, x2
    # else:
    #     return []

def findlastpoint(uboxes, boxes, low_u):
    print('len(bbox)', len(boxes))
    lastx = uboxes[1][2]
    if len(boxes) == 0:
        return lastx, low_u
    else:
        dist = []
        for b in boxes:
            if uboxes[1][2] - (b[2]+b[0])/2.0 >= 0:
                dist.append(uboxes[1][2] - (b[2]+b[0])/2.0)
        print('dddddddlast',dist)
        index = dist.index(min(dist))
        if 80< dist[index] <= 110:
            lastx = boxes[index][2]
            low_u += 1
            print('last: ', low_u)
        elif 0 <= dist[index] <= 80:
            lastx = boxes[index][2]

    return lastx, low_u

def findfirstpoint(uboxes, boxes):
    print('len(bbox)', len(boxes))
    firstx = uboxes[0][2] - 10
    if len(boxes) == 0:
        return firstx
    else:
        # dist =[b[0]-uboxes[0][2] if b[0]-uboxes[0][2] >= 0 else 1000 for b in boxes]
        dist = [abs(b[0] - uboxes[0][2]) for b in boxes]
        print('dddddddfirst',dist)
        if dist.count(min(dist)) == 1:
            index = dist.index(min(dist))
        elif dist.count(min(dist)) == 2:
            index = dist.index(min(dist)) + 1
        print('index', index)
        # print(index, boxes[index][0])
        if dist[index] <= 35:
            firstx = boxes[index][0]

    return firstx

def detectU(im, boxes, utags, umasks, uboxes, im_name, DEBUG):
    if utags:
        detect = detect_tags(type_tag='u', ratio=0.5, thresh_w=[16, 45], thresh_h=[47, 73], count=2, DEBUG=DEBUG,
                             DEBUG_DIR=DEBUG_DIR)
        u_num, switch = detect.detect_num(utags, im_name, umasks)
    else:
        return False, [], [], [], []
    print('u_num:', u_num)
    if u_num:
        set_unum, uboxes = selectu(u_num, uboxes)
    else:
        return False, [], [], [], []
    print('set_unum:', set_unum)
    # print('uboxes:',uboxes)

    ok = True
    up_u = 0
    low_u = 0
    low_u_new = 0
    if len(set_unum) == 2 and len(uboxes) == 2:
        up_u = int(set_unum[0])
        low_u = int(set_unum[1])

        firstx = findfirstpoint(uboxes, boxes)
        lastx, low_u_new = findlastpoint(uboxes, boxes, low_u)
        print('uboxes:', uboxes[0][2], 'first:', firstx, 'last', lastx)
        region = im[firstx:lastx, :, :]

        up_point = firstx
        low_point = lastx
        if DEBUG:
            result_image_path = os.path.join(DEBUG_DIR, im_name + "_jigui.jpg")
            cv2.imwrite(result_image_path, region)
    elif len(utags) <= 2:
        ok = 'TF'
        up_point = 0
        low_point = 0
    else:
        ok = False
        up_point = 0
        low_point = 0
        # print('detect u incorrectly')

    return ok, up_u, low_u, low_u_new, up_point, low_point

def findalltags(im, im_name, DEBUG):
    print('start find all tags..........................')
    lower_hue_low = [20, 60, 60]
    lower_hue_high = [32, 255, 255]

    hsv_image = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    kernel_size = (5, 5)
    mask_lower= create_hue_mask(hsv_image, lower_hue_low, lower_hue_high, kernel_size)
    if DEBUG:
        result_image_path = os.path.join(DEBUG_DIR, im_name + "_tags.jpg")
        cv2.imwrite(result_image_path, mask_lower)
    labels = measure.label(mask_lower, connectivity=2)
    pro = measure.regionprops(labels)

    # find all tags
    tagboxes = []
    tagimages = []
    tagmasks = []
    uimages = []
    uboxes = []
    umasks = []
    i = 0
    im_copy = im.copy()
    for p in pro:
        (x1, y1, x2, y2) = p.bbox
        #print('tagcccccc:', (x1, y1, x2, y2), (y2 - y1), (x2 - x1), p.area*1.0/((x2-x1)*(y2-y1)))
        if 230 >= (y2-y1) >= 100 and 40 >= (x2-x1) >= 20 and p.area*1.0/((x2-x1)*(y2-y1)) >= 0.6:
            #print(p.area*1.0/((x2-x1)*(y2-y1)))
            tagboxes.append(p.bbox)
        if 230 >= (y2-y1) >= 100 and 75 >= (x2-x1) > 40 and 0.3 < p.area * 1.0/((x2-x1) * (y2-y1)) < 0.9:
            print('tag width!!!!!!!!!!!!!!!!!!!!!!!')
            x1, x2 = findminbox(mask_lower[x1:x2, y1:y2], x1, x2)
            #print('x1,x2', x1, x2)
            if 51 >= (x2-x1) >= 20:
                tagboxes.append((x1, y1, x2, y2))
        if 50 <= y2-y1 <= 90 and 30 <= x2-x1 <= 65 and 2 > (y2-y1)*1.0/(x2-x1) > 1 and p.area*1.0/((x2-x1)*(y2-y1)) >= 0.65:
            i += 1
            uboxes.append(p.bbox)
            x = 0 if x1-1 <= 0 else x1-1
            y = 0 if y1-1 <= 0 else y1-1
            uimages.append(im[x:x2 + 1, y:y2 + 1, :])
            umasks.append(mask_lower[x:x2 + 1, y:y2 + 1])
            # uimages.append(im[x1:x2, y1:y2, :])
            # umasks.append(mask_lower[x1:x2, y1:y2])
            cv2.rectangle(im_copy, (y1, x1), (y2, x2), (0, 0, 255), 3)
            print('u:', y2 - y1, x2 - x1, p.area * 1.0 / ((x2 - x1) * (y2 - y1)))
    
    '''
    if len(uboxes) == 4:
        uboxes.sort(key=lambda x:x[1])
        left = uboxes[:2]
        right = uboxes[2:4]
        left.sort(key=lambda x:x[0])
        right.sort(key=lambda x:x[0])
        p0 = left[0]
        p1 = left[1]
        p2 = right[1]
        p3 = right[0]
        print(p0, p1, p2, p3)
        src_point = np.float32([[p0[1],p0[0]], [p1[1],p1[2]], [p2[3],p2[2]], [p3[3],p3[0]]])
        print(src_point)
        dsize=(1200, 800)
        dst_point = np.float32([[0,0],[0,dsize[1]-1],[dsize[0]-1,dsize[1]-1],[dsize[0]-1,0]])
        h, s = cv2.findHomography(src_point, dst_point, cv2.RANSAC, 5)
        im_trans = cv2.warpPerspective(im, h, dsize)
        cv2.imwrite('code/trans.jpg', im_trans)

        m = cv2.getPerspectiveTransform(src_point, dst_point)
        im_per = cv2.warpPerspective(im, m, dsize, flags=cv2.INTER_LINEAR)
        cv2.imwrite('code/pre.jpg', im_per)
    '''


    for i, box in enumerate(tagboxes):
        (x1, y1, x2, y2) = box
        print('tag:', (x1, y1, x2, y2), (y2 - y1), (x2 - x1))
        x1 = 0 if x1 - 1 < 0 else x1 - 1
        x2 = im.shape[0] if x2 + 1 > im.shape[0] else x2 + 1
        tagimages.append(im[x1:x2, y1-1:y2+1, :])
        tagmasks.append(mask_lower[x1:x2, y1-1:y2+1])
        # tagimages.append(im[x1:x2, y1:y2, :])
        # tagmasks.append(mask_lower[x1:x2, y1:y2])
        if DEBUG:
            result_image_path = os.path.join(DEBUG_DIR, im_name+'_'+str(i)+'_'+'tag.jpg')
            # cv2.imwrite(result_image_path, im[x1:x2, y1-5:y2+5, :])
            cv2.imwrite(result_image_path, im[x1:x2, y1:y2, :])
        cv2.rectangle(im_copy, (y1, x1), (y2, x2), (0, 0, 255), 3)
    if DEBUG:
        result_image_path = os.path.join(DEBUG_DIR, im_name + ".jpg")
        cv2.imwrite(result_image_path, im_copy)

    return tagimages, tagmasks, tagboxes, uimages, umasks, uboxes


def detecting(im_url, map1, map2, debug=None):
    start = time.time()
    DEBUG = debug
    im_name = im_url.split('/')[-1].split('.')[0]
    im = cv2.imread(im_url)
    
    image = np.zeros((1300, 2200, 3),dtype=np.uint8)
    for i in range(1300):
        for j in range(2200):
            image[i, j] = [255, 255, 255]
    
    for i in range(110, 1190):
        for j in range(140, 2060):
            image[i,j,:] = im[i-110, j-140,:]
    image = image.astype(np.uint8)
    im = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    #im = undistort(im)
    im_copy = im.copy()

    # save path of result image 
    image_file = os.path.join(config.DETECT_IMAGE_PATH, im_name + '.jpg')
    #image_file = os.path.join('code/result', im_name + '.jpg')

    # find all tags
    tagimages, tagmasks, boxes, uimages, umasks, uboxes = findalltags(im, im_name, DEBUG)

    # detect u tags
    u_range = []
    ok = True
    if len(uboxes) > 0:
        print('start detect U..................')
        ok, up_u, low_u, low_u_new, up_point, low_point = detectU(im, boxes, uimages, umasks, uboxes, im_name, DEBUG)
        u_range = [low_u, up_u]
        print('detect u result: ', ok, up_point, low_point)
    else:
        print('no u tags')
        up_u = 0
        low_u = 0 
        up_point = 0
        low_point = 0

    # detect equipment
    light_ok = True
    light_u = []
    final_result = []
    empty = True

    if len(uboxes) == 0 and len(boxes) == 0:
        cv2.imwrite(image_file, im)
        return False, final_result, image_file, u_range, light_ok, light_u
    
    if len(boxes) > 0 and ok != False:
        print('start detect all tags..................')
        detect = detect_tags(type_tag='ip', ratio=0.65, thresh_w=[16, 60], thresh_h=[34, 75], count=[], DEBUG=DEBUG,
                             DEBUG_DIR=DEBUG_DIR)
        result, result_switch = detect.detect_num(tagimages, im_name, tagmasks)
        print(len(boxes), len(result), len(result_switch))
        # visulize SWITCH
        if result_switch:
            print('display switch information')
            tagboxes = []
            for ind, res in result_switch:
                print(up_point, boxes[ind][0], boxes[ind][2], low_point+10, len(uboxes))
                tagboxes.append(boxes[ind])
                if (up_point <= boxes[ind][0] and boxes[ind][2] <= low_point+10) or (len(uboxes) <= 2 and low_u == up_u):
                    print('111111111111111111111')
                    if len(res) == 3:
                        u_index = res[1] + '~' + res[2]
                        u = (int(res[1]), int(res[2]))
                    else:
                        u_index = res[1]
                        u = (int(res[1]), int(res[1]))
                    cv2.putText(im_copy, 'IP: ' + res[0] + ' U: ' + u_index, (boxes[ind][1], boxes[ind][0]),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    final_result.append({'IP': res[0], 'U': u})      
                    empty = False

            if tagboxes:
                for box in tagboxes:
                    boxes.remove(box)

        # visulize IP
        if result:
            print('display ip information')
            print(len(boxes), len(result))
            for b, res in zip(boxes, result):
                if up_point <= b[0] and b[2] <= low_point+10 and res:
                    continue
                else:
                    boxes.remove(b)
                    result.remove(res)
            print(len(boxes), len(result))
            
            if ok == 'TF' and final_result==[]:
                ok = False
            else:
                ok = True
            '''
            # print(boxes)
            if result and boxes:
                count =len(result)-1 if len(result)%2 != 0 else len(result)
                for ind in range(0, count, 2):
                    res1 = result[ind + 1]
                    res2 = result[ind]
                    res = res1 if len(res1) >= len(res2) else res2
                    # print(boxes[ind][0], boxes[ind + 1][2])
                    end_u = int(round(up_u - (boxes[ind][0] - up_point) / height_u) - 1)
                    start_u = int(round(up_u - (boxes[ind + 1][2] - up_point) / height_u))
                    # print(up_u - (boxes[ind][0] - up_point) / height_u - 1, up_u - (boxes[ind + 1][2] - up_point) / height_u)
                    u_list = np.arange(start_u, end_u + 1)
                    cv2.putText(im_copy, 'IP: {}, U: {}'.format(res, u_list),
                                (boxes[ind][1], boxes[ind][0] + 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    final_result.append({'IP': res, 'U': (start_u, end_u)})

            '''
           
            if result and boxes:
                # detect red light 
    		light_ok, light_locat = detect_redlight(im, im_name, DEBUG)
    		if low_u != up_u:
        	    sum_u = up_u - low_u_new
                    height_u = (low_point - up_point) * 1.0 / sum_u
                    for light in light_locat:
                        if boxes[0][0] <= light[1] <= low_point:
                            print('light: ', light[3])
                            curr_u = int(up_u - np.ceil((light[1] - up_point) / height_u))
            		    light_u.append(curr_u)
            		    cv2.circle(im_copy, (light[0], light[1]), light[2], (0,0,255), 5)
            		    cv2.putText(im_copy, 'U: {}'.format(curr_u), (light[0], light[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                empty = False
              
                for ind in range(len(boxes)):
                    res = result[ind]
                    count_u = ((boxes[ind][2]+boxes[ind][0])/2.0 - up_point) / height_u
                    if count_u-np.floor(count_u) < 0.1:
                        curr_u = int(round(up_u - np.floor(count_u)))
                    else:
                        curr_u = int(round(up_u - np.ceil(count_u)))
                    #count_u =  (boxes[ind][2] - up_point) / height_u
                    #if count_u-np.floor(count_u) > 0.3:
                    #    curr_u = int(up_u - np.ceil(count_u))
                    #else:
                    #    curr_u = int(up_u - np.floor(count_u))
                    #print(count_u)
                    print(((boxes[ind][2]+boxes[ind][0])/2.0 - up_point) / height_u)
                    cv2.putText(im_copy, 'IP: {}, U: {}'.format(res, curr_u),
                                (boxes[ind][1], boxes[ind][0] + 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    final_result.append({'IP': res, 'U': curr_u})

    if empty and ok:
        cv2.putText(im_copy, 'NULL', (500, 500), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 3)
    elif not ok:
        ok = False
        cv2.putText(im_copy, 'False', (500, 500), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 3)
    cv2.imwrite(image_file, im_copy)
    end = time.time()
    print('spend time is: ', end-start)
    return ok, final_result, image_file, u_range, light_ok, light_u
