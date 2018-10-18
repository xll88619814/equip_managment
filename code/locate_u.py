import os
import cv2
import sys
import time
import numpy as np
from skimage import measure
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
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        dilated = cv2.dilate(mask, kernel)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 13))
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
        if len(u) > 0:
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
        set_unum = boxes = []

    return set_unum, boxes

def findminbox(im, x1, x2):
    #cv2.imshow('box', im)
    #cv2.waitKey(0)
    w = im.shape[1]
    ratio = np.sum(im == 255)*1.0/((x2-x1)*w)
    #print(ratio)
    while ratio < 0.65:
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
        #print(ratio)

    #cv2.imshow('box', im)
    #cv2.waitKey(0)
    #print(x1,x2, im.shape)
    return x1, x2
    # else:
    #     return []



def findfirstpoint(uboxes, boxes):
    print('len(bbox)', len(boxes))
    firstx = uboxes[0][2] - 10
    #lastx = uboxes[1][2]
    if len(boxes) == 0:
        return firstx
    else:
        #dist =[b[0]-uboxes[0][2] if b[0]-uboxes[0][2] >= 0 else 1000 for b in boxes]
        dist = [abs(b[0] - uboxes[0][2]) for b in boxes]
        print('ddddddd',dist)
        if dist.count(min(dist)) == 1:
            index = dist.index(min(dist))
        elif dist.count(min(dist)) == 2:
            index = dist.index(min(dist)) +1
        print('index', index)
        #print(index, boxes[index][0])
        if dist[index] < 50:
            firstx = boxes[index][0]

    return firstx

def detectU(im, boxes, utags, umasks, uboxes, im_name, DEBUG):
    if utags == []:
        return False, [], [], [], []
    else:
        detect = detect_tags(type_tag='u', ratio=0.5, thresh_w=[16, 46], thresh_h=[48, 80], count=2, DEBUG=DEBUG, DEBUG_DIR=DEBUG_DIR)
        u_num, switch = detect.detect_num(utags, im_name, umasks)
    print('u_num:', u_num)
    if u_num == []:
        return False, [], [], [], []
    else:
        set_unum, uboxes = selectu(u_num, uboxes)
    print('set_unum:', set_unum)
    # print('uboxes:',uboxes)

    ok = True
    up_u = 0
    low_u = 0
    region = im.copy()
    # print('jigui tags is: ', len(uboxes))
    # adjust jigui redion

    if len(set_unum) == 2 and len(uboxes) == 2:
        up_u = int(set_unum[0])
        low_u = int(set_unum[1])

        firstx = findfirstpoint(uboxes, boxes)
        print('uboxes:', uboxes[0][2], 'first:', firstx)
        region = im[firstx:uboxes[1][2], :, :]

        up_point = firstx
        low_point = uboxes[1][2]
        if DEBUG:
            result_image_path = os.path.join(DEBUG_DIR, im_name + "_jigui.jpg")
            cv2.imwrite(result_image_path, region)
    else:
        ok = False
        up_point = 0
        low_point = 0
        # print('detect u incorrectly')

    return ok, up_u, low_u, up_point, low_point

def findalltags(im, im_name, DEBUG):
    print('start detect switch tags..........................')
    lower_hue_low = [20, 90, 65]
    lower_hue_high = [30, 255, 255]

    hsv_image = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    kernel_size = (5, 5)
    mask_lower= create_hue_mask(hsv_image, lower_hue_low, lower_hue_high, kernel_size)
    if DEBUG:
        result_image_path = os.path.join(DEBUG_DIR, im_name + "_switch.jpg")
        cv2.imwrite(result_image_path, mask_lower)
    labels = measure.label(mask_lower, connectivity=2)
    pro = measure.regionprops(labels)

    # find all tags
    switchboxes = []
    switchtags = []
    switchmasks = []
    utags = []
    uboxes = []
    umasks = []
    i = 0
    im_copy = im.copy()
    for p in pro:
        (x1, y1, x2, y2) = p.bbox
        if 230 >= (y2-y1) >= 100 and 40 >= (x2-x1) >= 20 and p.area*1.0/((x2-x1)*(y2-y1)) >= 0.6:
            print(p.area*1.0/((x2-x1)*(y2-y1)))
            switchboxes.append(p.bbox)
        if 230 >= (y2-y1) >= 100 and 75 >= (x2-x1) > 40 and 0.3 < p.area * 1.0/((x2-x1) * (y2-y1)) < 0.6:
            print('tag width!!!!!!!!!!!!!!!!!!!!!!!')
            x1, x2 = findminbox(mask_lower[x1:x2, y1:y2], x1, x2)
            if 40 >= (x2-x1) >= 20:
                switchboxes.append((x1, y1, x2, y2))
        if 30 <= y2-y1 <= 90 and 30 <= x2-x1 <= 65 and 2 > (y2-y1)*1.0/(x2-x1) > 1 and p.area*1.0/((x2-x1)*(y2-y1)) >= 0.7:
            i += 1
            x = 0 if x1-5 <= 0 else x1-5
            y = 0 if y1-5 <= 0 else y1-5
            utags.append(im[x:x2 + 5, y:y2 + 5, :])
            uboxes.append(p.bbox)
            umasks.append(mask_lower[x:x2 + 5, y:y2 + 5])
            cv2.rectangle(im_copy, (y1, x1), (y2, x2), (0, 0, 255), 3)
            print('u:', y2 - y1, x2 - x1, p.area * 1.0 / ((x2 - x1) * (y2 - y1)))


    for i, box in enumerate(switchboxes):
        (x1, y1, x2, y2) = box
        print('switch:', (x1, y1, x2, y2), (y2 - y1), (x2 - x1))
        x1 = 0 if x1 - 5 < 0 else x1 - 5
        x2 = im.shape[0] if x2 + 5 > im.shape[0] else x2 + 5
        switchtags.append(im[x1:x2, y1-5:y2+5, :])
        switchmasks.append(mask_lower[x1:x2, y1-5:y2+5])
        if DEBUG:
            result_image_path = os.path.join(DEBUG_DIR, im_name+'_'+str(i)+'_'+'switch.jpg')
            cv2.imwrite(result_image_path, im[x1:x2, y1-5:y2+5, :])
        cv2.rectangle(im_copy, (y1, x1), (y2, x2), (0, 0, 255), 3)
    if DEBUG:
        result_image_path = os.path.join(DEBUG_DIR, im_name + ".jpg")
        cv2.imwrite(result_image_path, im_copy)

    return switchtags, switchmasks, switchboxes, utags, umasks, uboxes


def findIPregion(img, im, up_u, low_u, u_point, lower_hue_low, lower_hue_high, im_name, DEBUG):
    final_result = []
    sum_u = up_u - low_u
    height_u = img.shape[0]*1.0/sum_u

    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    kernel_size = (5, 5)
    mask_lower = create_hue_mask(hsv_image, lower_hue_low, lower_hue_high, kernel_size)
    if DEBUG:
        result_image_path = os.path.join(DEBUG_DIR, im_name + "_equipment.jpg")
        cv2.imwrite(result_image_path, mask_lower)

    labels = measure.label(mask_lower, connectivity=2)
    pro = measure.regionprops(labels)

    # remain good IP regions
    min_width = 1000
    for p in pro:
        (x1, y1, x2, y2) = p.bbox
        width = y2 - y1
        if 250 >= width >= 150 and 60 >= x2-x1 >= 25:
            if min_width > width:
                min_width = width
    print('min_width:', min_width)

    box = []
    im_copy = img.copy()
    for p in pro:
        (x1, y1, x2, y2) = p.bbox
        if min_width <= y2-y1 <= min_width + 30 and x2-x1 >= 25:
            box.append(p.bbox)
            print('x2x1', y2-y1, x2-x1)
            cv2.rectangle(im_copy, (y1, x1), (y2, x2), (0, 0, 255), 3)
    cv2.imwrite(im_name+'.jpg', im_copy)
    print('len(box)', len(box))
    if len(box) % 2 != 0:
        box = box[:-1]
    if len(box) > 0 and len(box) % 2 == 0:
        iptags = []
        ipboxes = []
        ipmasks = []
        u = []
        for i, b in enumerate(box):
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
            if i % 2 == 0:
                end_u = int(round(up_u - b[0]/height_u) -1)
                start_u = int(round(up_u - box[i+1][2]/height_u))
                u.append((start_u, end_u))
                #print(end_u, start_u)

        detect = detect_tags(type_tag='ip', ratio=0.6, thresh_w=[15, 52], thresh_h=[40, 73], count=[], DEBUG=DEBUG, DEBUG_DIR=DEBUG_DIR)
        result, switch = detect.detect_num(iptags, im_name, ipmasks)

        if result:
            count = len(result)
            for ind in range(0, count, 2):
                res1 = result[ind]
                res2 = result[ind+1]
                res = res1 if len(res1) > len(res2) else res2
                u_list = np.arange(u[ind/2][0], u[ind/2][1]+1)
                cv2.putText(im, 'IP: {}, U: {}'.format(res, u_list), (ipboxes[ind][1], u_point+ipboxes[ind][0]+80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                final_result.append({'IP': res, 'U': u[ind/2]})

    return final_result, im


def detecting(im_url, debug=None):
    start = time.time()
    im_name = im_url.split('/')[-1].split('.')[0]
    im = cv2.imread(im_url)

    image_file = os.path.join(config.DETECT_IMAGE_PATH, im_name + '.jpg')
    #image_file = os.path.join('code/result', im_name + '.jpg')

    DEBUG = debug
    final_result = []

    # judge if it is switch
    switchtags, switchmasks, boxes, utags, umasks, uboxes = findalltags(im, im_name, DEBUG)

    ok = True
    if len(boxes) > 0:
        print('start detect U..................')
        ok, up_u, low_u, up_point, low_point = detectU(im, boxes, utags, umasks, uboxes, im_name, DEBUG)
        print(up_point, low_point)

        detect = detect_tags(type_tag='switch', ratio=0.65, thresh_w=[18, 65], thresh_h=[45, 85], count=[], DEBUG=DEBUG,
                             DEBUG_DIR=DEBUG_DIR)
        result, result_switch = detect.detect_num(switchtags, im_name, switchmasks)

        # visulize
        if result_switch:
            switchboxes = []
            for ind, res in result_switch:
                if (up_point <= boxes[ind][0] and boxes[ind][0] <= low_point and ok) or not ok:
                    if len(res) == 3:
                        u_index = res[1]+'~'+res[2]
                    else:
                        u_index = res[1]
                    cv2.putText(im, 'IP: ' + res[0] + ' U: ' + u_index, (boxes[ind][1], boxes[ind][0]),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    final_result.append({'IP': res[0], 'U': [res[1:]]})
                    switchboxes.append(boxes[ind])
            if switchboxes:
                for box in switchboxes:
                    boxes.remove(box)

        if ok and result:
            print(len(boxes), len(result))
            for b, res in zip(boxes, result):
                if up_point <= b[0] and b[2] <= low_point+10:
                    continue
                else:
                    boxes.remove(b)
                    result.remove(res)

            sum_u = up_u - low_u
            height_u = (low_point - up_point) * 1.0 / sum_u
            print(boxes)
            if result and boxes:
                count =len(result)-1 if len(result)%2 != 0 else len(result)
                for ind in range(0, count, 2):
                    res1 = result[ind + 1]
                    res2 = result[ind]
                    res = res1 if len(res1) >= len(res2) else res2
                    print(boxes[ind][0], boxes[ind + 1][2])
                    end_u = int(round(up_u - (boxes[ind][0] - up_point) / height_u) - 1)
                    start_u = int(round(up_u - (boxes[ind + 1][2] - up_point) / height_u))
                    #print(up_u - (boxes[ind][0] - up_point) / height_u - 1, up_u - (boxes[ind + 1][2] - up_point) / height_u)
                    u_list = np.arange(start_u, end_u + 1)
                    cv2.putText(im, 'IP: {}, U: {}'.format(res, u_list),
                                (boxes[ind][1], boxes[ind][0] + 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    final_result.append({'IP': res, 'U': (start_u, end_u)})

    cv2.imwrite(image_file, im)
    end = time.time()
    print('spend time is: ', end-start)
    return ok, final_result, image_file
