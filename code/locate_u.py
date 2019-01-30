import os
import cv2
import time
import json
import numpy as np
from skimage import measure
#from fisheye import undistort
from warptrans import transimage
from detectnum import detect_tags
from houghtrans import houghtrans
from detectlight import detect_light
from config import config
from db_connecttion.MySqlConn_gxxj import Mysql

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

def get_bean(detectsetid):
    mysql = Mysql()
    sql = "SELECT detect_setting FROM rb_detect_setting WHERE detect_setting_id = %s"
    param = []
    param.append(detectsetid)
    result = mysql.getOne(sql, param)
    if result:
        data = result["detect_setting"]
        mysql.dispose()
        return data
    else:
        return False

def analyze_data(datas):
    datas = json.loads(datas)
    rows = len(datas)
    print('rows:', rows)
    LIGHT = False
    EQUIP = False
    for row in datas:
        print('raw:', raw)
        if row['type'] == '8':
            EQUIP = True
        elif row['type'] == '1':
            LIGHT = True
    return LIGHT, EQUIP

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

def findminbox(im, box, thresh_ratio, w_thresh, h_thresh):
    print('tag !!!!!!!!!!!!!!!!!!!!!!!')
    h, w = im.shape
    ratio = np.sum(im == 255)*1.0/(h*w)
  
    x1, y1, x2, y2 = box
    while ratio < thresh_ratio:
        h, w = im.shape
        if x2-x1 < h_thresh or y2-y1 < w_thresh:
            return False, 0, 0, 0, 0
        ratio1 = np.sum(im[2:, :] == 255) * 1.0 / ((x2-x1-1) * w)
        ratio2 = np.sum(im[:-2, :] == 255) * 1.0 / ((x2-x1-1) * w)
        ratio3 = np.sum(im[:, 2:] == 255) * 1.0 / ((y2-y1-1) * h)
        ratio4 = np.sum(im[:, :-2] == 255) * 1.0 / ((y2-y1-1) * h)
        ratio_max = np.max([ratio1, ratio2, ratio3, ratio4])
        if ratio_max == ratio1:
            im = im[2:, :]
            ratio = ratio1
            x1 += 2
        elif ratio_max == ratio2:
            im = im[:-2, :]
            ratio = ratio2
            x2 -= 2
        elif ratio_max == ratio3:
            im = im[:, 2:]
            ratio = ratio3
            y1 += 2
        elif ratio_max == ratio4:
            im = im[:, :-2]
            ratio = ratio2
            y2 -= 2
        print(ratio, x2-x1, y2-y1, [ratio1, ratio2, ratio3, ratio4])

    return True, x1, y1, x2, y2


def findminbox_x(im, x1, x2, thresh_ratio):
    print('IP tag hight!!!!!!!!!!!!!!!!!!!!!!!')
    w = im.shape[1]
    ratio = np.sum(im == 255)*1.0/((x2-x1)*w)

    x1_copy = x1
    x2_copy = x2
    while ratio < thresh_ratio:
        if x2-x1 < 20:
            return False, x1_copy, x2_copy
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
        print(ratio, x2-x1)

    return True, x1, x2

def findminbox_y(im, x1, x2, thresh_ratio):
    print('IP tag width!!!!!!!!!!!!!!!!!!!!!!!')
    w = im.shape[0]
    ratio = np.sum(im == 255)*1.0/((x2-x1)*w)

    while ratio < thresh_ratio:
        if x2-x1 < 30:
            return 0, 0
        ratio1 = np.sum(im[:, 2:] == 255) * 1.0 / ((x2-x1-1) * w)
        ratio2 = np.sum(im[:, :-2] == 255) * 1.0 / ((x2-x1-1) * w)
        if ratio1 > ratio2:
            im = im[:, 2:]
            ratio = ratio1
            x1 += 2
        elif ratio1 < ratio2:
            im = im[:, :-2]
            ratio = ratio2
            x2 -= 2
        else:
            im = im[:, 2:-2]
            ratio = ratio1
            x1 += 2
            x2 -= 2
        print(ratio, x2-x1)

    return x1, x2

def findlastpoint(uboxes, boxes, low_u, angle):
    print('len(bbox)', len(boxes))
    lastx = uboxes[1][2]
    if len(boxes) == 0:
        return lastx, low_u
    else:
        dist = []
        for b in boxes:
            if uboxes[1][2] - (b[2]+b[0])/2.0 >= 0:
                dist.append(uboxes[1][2] - (b[2]+b[0])/2.0)
        print('dddddddlast',uboxes[1][2], dist)
        index = dist.index(min(dist))
        if 74 <= dist[index] <= 100:
            lastx = boxes[index][2]
            low_u += 1
            print('last: ', low_u)
        #elif 40 <= dist[index] <= 70:
            #lastx = boxes[index][2]
        elif 0 <= dist[index] <= 40:
            lastx = boxes[index][2]

    return lastx, low_u

def findfirstpoint(uboxes, boxes, angle):
    print('len(bbox)', len(boxes))
    firstx = uboxes[0][2] - 10
    if angle == 0:
        dist_thresh = 30
    else:
        dist_thresh = 42
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
        if dist[index] <= dist_thresh:
            firstx = boxes[index][0]

    return firstx

def detectU(im, boxes, utags, umasks, uboxes, im_name, angle, DEBUG):
    if utags:
        detect = detect_tags(type_tag='u', ratio=0.5, thresh_w=[13, 45], thresh_h=[43, 72], count=2, DEBUG=DEBUG,
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

        firstx = findfirstpoint(uboxes, boxes, angle)
        lastx, low_u_new = findlastpoint(uboxes, boxes, low_u, angle)
        print('uboxes:', uboxes[0][2], 'first:', firstx, 'last', lastx)
        region = im[firstx:lastx, :, :]

        up_point = firstx
        low_point = lastx
        if DEBUG:
            result_image_path = os.path.join(DEBUG_DIR, im_name + "_jigui.jpg")
            cv2.imwrite(result_image_path, region)
    elif len(utags) <= 2 and len(set_unum) < 2:
        ok = 'TF'
        up_point = 0
        low_point = 0
    elif len(utags) > 2 and len(set_unum) < 2:
        ok = False
        up_point = 0
        low_point = 0
        # print('detect u incorrectly')

    return ok, up_u, low_u, low_u_new, up_point, low_point

def detectIP(im, ok, tagimages, tagmasks, boxes, uboxes, up_u, low_u_new, low_point, up_point, im_name, DEBUG):
    print('start detect all tags..................')
    detect = detect_tags(type_tag='ip', ratio=0.65, thresh_w=[16, 60], thresh_h=[34, 75], count=[], DEBUG=DEBUG,
                         DEBUG_DIR=DEBUG_DIR)
    result, result_switch = detect.detect_num(tagimages, im_name, tagmasks)
    print(len(boxes), len(result), len(result_switch))

    
    empty = True
    final_result = []
    # visulize SWITCH
    if result_switch:
        print('display switch information')
        tagboxes = []
        for ind, res in result_switch:
            print(up_point, boxes[ind][0], boxes[ind][2], low_point+10, len(uboxes))
            tagboxes.append(boxes[ind])
            if (up_point <= boxes[ind][0] and boxes[ind][2] <= low_point+10) or (len(uboxes) <= 2 and low_u_new == up_u):
                if len(res) == 3:
                    u_index = res[1] + '~' + res[2]
                    u = (int(res[1]), int(res[2]))
                else:
                    u_index = res[1]
                    u = (int(res[1]), int(res[1]))
                cv2.putText(im, 'IP: ' + res[0] + ' U: ' + u_index, (boxes[ind][1], boxes[ind][0]),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                final_result.append({'IP': res[0], 'U': u})      
                empty = False

        if tagboxes:
            for box in tagboxes:
                boxes.remove(box)

    # the number of u is less than two and is not switch
    if up_u == low_u_new and empty:
        ok = False

    if ok == 'TF' and final_result == []:
        ok = False
    elif ok == 'TF' and final_result != []:
        ok = True

    # visulize IP
    if result and ok and low_u_new != up_u:
        sum_u = up_u - low_u_new
        height_u = (low_point - up_point) * 1.0 / sum_u
        print('display ip information')
        for b, res in zip(boxes, result):
            if up_point <= b[0] and b[2] <= low_point+10 and res:
                continue
            else:
                boxes.remove(b)
                result.remove(res)
        print(len(boxes), len(result))
        
        if boxes:
            empty = False
            for ind in range(len(boxes)):
                res = result[ind]
                count_u = ((boxes[ind][2]+boxes[ind][0])/2.0 - up_point) / height_u
                if count_u-np.floor(count_u) < 0.01:
                    curr_u = int(round(up_u - np.floor(count_u)))
                else:
                    curr_u = int(round(up_u - np.ceil(count_u)))
                print(((boxes[ind][2]+boxes[ind][0])/2.0 - up_point) / height_u)
                cv2.putText(im, 'IP: {}, U: {}'.format(res, curr_u), (boxes[ind][1], boxes[ind][0] + 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                final_result.append({'IP': res, 'U': curr_u})

    return ok, im, final_result, empty


def findalltags(im, im_name, DEBUG):
    print('start find IP tags..........................')
    lower_hue_low = [23, 100, 100]
    lower_hue_high = [32, 255, 255]

    hsv_image = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    kernel_size = (5, 5)
    mask_lower= create_hue_mask(hsv_image, lower_hue_low, lower_hue_high, kernel_size)
    if DEBUG:
        result_image_path = os.path.join(DEBUG_DIR, im_name + "_IPtags.jpg")
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
    
    im_copy = im.copy()
    for p in pro:
        (x1, y1, x2, y2) = p.bbox
        #print('tagcccccc:', (x1, y1, x2, y2), (y2 - y1), (x2 - x1), p.area*1.0/((x2-x1)*(y2-y1)))
        if 230 >= (y2-y1) >= 100 and 40 >= (x2-x1) >= 20 and p.area*1.0/((x2-x1)*(y2-y1)) >= 0.58:
            print('tag:', (x1, y1, x2, y2), (y2 - y1), (x2 - x1), p.area*1.0/((x2-x1)*(y2-y1)))
            tagboxes.append(p.bbox)
        if 230 >= (y2-y1) >= 100 and 75 >= (x2-x1) > 40 and 0.4 < p.area * 1.0/((x2-x1) * (y2-y1)) < 0.9:
            ok, x1, y1, x2, y2 = findminbox(mask_lower[x1:x2, y1:y2], p.bbox, 0.7, 100, 20)
            if 51 >= (x2-x1) >= 20 and ok:
                print('tag:', (x1, y1, x2, y2), (y2 - y1), (x2 - x1), p.area*1.0/((x2-x1)*(y2-y1)))
                tagboxes.append((x1, y1, x2, y2))

    for i, box in enumerate(tagboxes):
        (x1, y1, x2, y2) = box
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


    print('start find U tags...........................')
    lower_hue_low = [23, 55, 45]
    lower_hue_high = [33, 255, 255]

    hsv_image = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    kernel_size = (5, 5)
    mask_lower= create_hue_mask(hsv_image, lower_hue_low, lower_hue_high, kernel_size)
    if DEBUG:
        result_image_path = os.path.join(DEBUG_DIR, im_name + "_Utags.jpg")
        cv2.imwrite(result_image_path, mask_lower)
    labels = measure.label(mask_lower, connectivity=2)
    pro = measure.regionprops(labels)
    i = 0
    for p in pro:
        (x1, y1, x2, y2) = p.bbox
        if 41 <= y2-y1 <= 92 and 31 <= x2-x1 <= 45 and 2.3 > (y2-y1)*1.0/(x2-x1) > 0.9 and p.area*1.0/((x2-x1)*(y2-y1)) >= 0.65:
            print('u:', (x1, y1, x2, y2), y2 - y1, x2 - x1, p.area * 1.0 / ((x2 - x1) * (y2 - y1)))
        elif 45 <= y2-y1 <= 95 and 45 < x2-x1 <= 75 and 2 > (y2-y1)*1.0/(x2-x1) > 0.6 and 1 > p.area*1.0/((x2-x1)*(y2-y1)) >= 0.4:
            #if y2-y1 > 60:
            #    y1, y2 = findminbox_y(mask_lower[x1:x2, y1:y2], y1, y2, 0.85)
            #ok, x1, x2 = findminbox_x(mask_lower[x1:x2, y1:y2], x1, x2, 0.755)
          
            ok, x1, y1, x2, y2 = findminbox(mask_lower[x1:x2, y1:y2], p.bbox, 0.85, 40, 30)
            print('find u ', ok, x2-x1, y2-y1)
            if 30 <= (x2-x1) <= 60:
                print('u:', (x1, y1, x2, y2), y2-y1, x2-x1, p.area * 1.0 / ((x2 - x1) * (y2 - y1)))
            elif (x2-x1) > 60:
                ok, x1, y1, x2, y2 = findminbox(mask_lower[x1:x2, y1:y2], (x1, y1, x2, y2), 0.95, 40, 30)
                if 30 <= (x2-x1) <= 60:
                    print('u:', (x1, y1, x2, y2), y2-y1, x2-x1, p.area * 1.0 / ((x2 - x1) * (y2 - y1)))   
                else:
                    continue
            else:
                continue           
        else:
            continue
        i += 1
        uboxes.append((x1, y1, x2, y2))
        x = 0 if x1-1 <= 0 else x1-1
        y = 0 if y1-1 <= 0 else y1-1
        uimages.append(im[x:x2 + 1, y:y2 + 1, :])
        umasks.append(mask_lower[x:x2 + 1, y:y2 + 1])
        # uimages.append(im[x1:x2, y1:y2, :])
        # umasks.append(mask_lower[x1:x2, y1:y2])
        cv2.rectangle(im_copy, (y1, x1), (y2, x2), (0, 0, 255), 3)
        
    if DEBUG:
        result_image_path = os.path.join(DEBUG_DIR, im_name + ".jpg")
        cv2.imwrite(result_image_path, im_copy)

    return tagimages, tagmasks, tagboxes, uimages, umasks, uboxes

def detecting(im_url, angle, detectsetid, debug=None):
    start = time.time()
    DEBUG = debug
    im_name = im_url.split('/')[-1].split('.')[0]
    im = cv2.imread(im_url)
    if get_bean(detectsetid):
        data = get_bean(detectsetid)
    else:
        return False, None, None, None, None, None
    LIGHT, EQUIP = analyze_data(data)
    #LIGHT = True
    #EQUIP = True
      
    im = houghtrans(im)
    im = transimage(im, float(angle))
    im_copy = im.copy()

    # save path of result image 
    image_file = os.path.join(config.DETECT_IMAGE_PATH, im_name + '.jpg')
    #image_file = os.path.join('code/result', im_name + '.jpg')

    # find U tags
    tagimages, tagmasks, boxes, uimages, umasks, uboxes = findalltags(im, im_name, DEBUG)

    # detect u tags
    u_range = []
    ok = True
    if len(uboxes) > 1:
        print('start detect U..................')
        ok, up_u, low_u, low_u_new, up_point, low_point = detectU(im, boxes, uimages, umasks, uboxes, im_name, float(angle), DEBUG)
        u_range = [low_u, up_u]
        print('detect u result: ', ok, up_point, low_point)
    else:
        print('no u tags')
        up_u = 0
        low_u = 0 
        low_u_new = 0
        up_point = 0
        low_point = 0

    light_ok = True
    light_u = []
    print(low_u,  up_u, boxes)
    if LIGHT and low_u != up_u and boxes:
        # detect light
        print('start detect light..................')
        im_copy, light_ok, light_u = detect_light(im, im_name, up_u, low_u_new, low_point, up_point, DEBUG, DEBUG_DIR)

    final_result = []
    empty = True
    if EQUIP and len(boxes) > 0 and ok != False:
        # detect equipment
        ok, im_copy, final_result, empty = detectIP(im_copy, ok, tagimages, tagmasks, boxes,  uboxes, up_u, low_u_new, low_point, up_point, im_name, DEBUG)
    
    print('ok: ', ok)
    if (len(boxes) == 0 or empty) and ok:
        cv2.putText(im_copy, 'NULL', (500, 500), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 3)
    elif not ok:
        cv2.putText(im_copy, 'False', (500, 500), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 3)

    cv2.imwrite(image_file, im_copy)
    end = time.time()
    print('spend time is: ', end-start)
    return ok, final_result, image_file, u_range, light_ok, light_u
