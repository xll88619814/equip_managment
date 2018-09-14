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

    boxes = []
    for p in pro:
        (x1, y1, x2, y2) = p.bbox
        if 380 >= (y2-y1) >= 150 and 55 >= (x2-x1) >= 25 and p.area*1.0/((x2-x1)*(y2-y1)) > 0.7:
            boxes.append(p.bbox)
    print('len(bbox)',len(boxes))
    #min_width = 1000
    #for p in pro:
    #    (x1, y1, x2, y2) = p.bbox
    #    width = y2 - y1
    #    if w_max > width > w_min and x2 - x1 >= 25:
    #        if min_width > width:
    #            min_width = width
    #
    #for p in pro:
    #    (x1, y1, x2, y2) = p.bbox
    #    if min_width <= y2-y1 <= min_width+30 and x1 >= uboxes[0][0] and x2 <= uboxes[1][2]+10:
    #        boxes.append(p.bbox)

    firstx = uboxes[0][2] - 10
    if len(boxes) == 0:
        return firstx
    else:
        dist = abs(boxes[0][0] - uboxes[0][2])
        if dist < 100:
            firstx = 0 if (boxes[0][0] - 5) <= 0 else (boxes[0][0] - 5)
            for p in boxes[1:]:
                (x1, y1, x2, y2) = p
                if abs(x1 - uboxes[0][2]) < dist:
                    dist = abs(x1 - uboxes[0][2])
                    firstx = x1 - 5
    return firstx

def findUregion(im, lower_hue_low, lower_hue_high, im_name, DEBUG):
    # find u region
    hsv_image = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    kernel_size = (10, 10)
    mask_lower = create_hue_mask(hsv_image, lower_hue_low, lower_hue_high, kernel_size)
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
        if 30 <= y2 - y1 <= 90 and 30 <= x2 - x1 <= 65 and 2 > (y2-y1)*1.0/(x2-x1) > 1 and p.area*1.0/((x2-x1)*(y2-y1)) > 0.7:  # and y2<0.9*im_width and y1>0.1*im_width and x1>0.1*im_height:
            i += 1
            x = 0 if x1-5 <= 0 else x1-5
            y = 0 if y1-5 <= 0 else y1-5
            utags.append(im[x:x2 + 5, y:y2 + 5, :])
            uboxes.append(p.bbox)
            umasks.append(mask_lower[x:x2 + 5, y:y2 + 5])
            cv2.rectangle(im_copy, (y1, x1), (y2, x2), (0, 0, 255), 3)
            #cv2.imwrite(im_name+'_mask_'+str(ind)+'.jpg', mask_lower[x:x2 + 5, y:y2 + 5])
            #cv2.imwrite(im_name+'_tags_'+str(ind)+'.jpg', im[x:x2 + 5, y:y2 + 5])
            print('u:', y2-y1, x2-x1, p.area*1.0/((x2-x1)*(y2-y1)))
    if DEBUG:
        result_image_path = os.path.join(DEBUG_DIR, im_name + ".jpg")
        cv2.imwrite(result_image_path, im_copy)
    # print('utags:',len(utags))

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

        firstx = findfirstpoint(im, uboxes, im_name, lower_hue_low, lower_hue_high, 150, 250, DEBUG)
        print('uboxes:', uboxes[0][2], 'first:', firstx)
        region = im[firstx:uboxes[1][2]-5, :, :]

        up_point = firstx
        low_point = uboxes[1][2]-5
        if DEBUG:
            result_image_path = os.path.join(DEBUG_DIR, im_name + "_jigui.jpg")
            cv2.imwrite(result_image_path, region)
    else:
        ok = False
        up_point = 0
        low_point = 0
        # print('detect u incorrectly')

    return ok, up_u, low_u, up_point, low_point

def isswitch(im, im_name, DEBUG):
    print('start detect switch tags..........................')
    switch = False
    server = False
    final_result = []
    lower_hue_low = [20, 90, 65]
    lower_hue_high = [31, 255, 255]

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
    boxes = []
    i = 0
    u_box = []
    for p in pro:
        (x1, y1, x2, y2) = p.bbox
        if 380 >= (y2-y1) >= 150 and 55 >= (x2-x1) >= 25 and p.area*1.0/((x2-x1)*(y2-y1)) > 0.7:
            x1 = 0 if x1 - 5 < 0 else x1 - 5
            x2 = im.shape[0] if x2 + 5 > im.shape[0] else x2 + 5
            print('switch:', (x1, y1, x2, y2), (y2-y1), (x2-x1))
            switchboxes.append(p.bbox)
            switchtags.append(im[x1:x2, y1-5:y2+5, :])
            switchmasks.append(mask_lower[x1:x2, y1-5:y2+5])
            if DEBUG:
                result_image_path = os.path.join(DEBUG_DIR, im_name+'_'+str(i)+'_'+'switch.jpg')
                cv2.imwrite(result_image_path, im[x1:x2, y1-5:y2+5, :])
            #cv2.imshow('switch', im[x1-5:x2+5,y1-5:y2+5,:])
            #cv2.waitKey(0)
            boxes.append(p.bbox)
            i += 1
        if 30 <= y2 - y1 <= 70 and 30 <= x2 - x1 <= 55 and 2 >= (y2-y1)*1.0/(x2-x1) >= 1 and p.area*1.0/((x2-x1)*(y2-y1)) > 0.7:
            u_box.append(p.bbox)

    if len(u_box) > 0:
        count_u = 1
        for ind, b in enumerate(u_box[1:]):
            (x1, y1, x2, y2) = b
            if (x1 - u_box[ind][0]) < 100:
                continue
            else:
                count_u += 1
    else:
        count_u = 0

    if len(switchboxes) > 0:
        detect = detect_tags(type_tag='switch', ratio=0.65, thresh_w=[15, 60], thresh_h=[40, 74], count=[], DEBUG=DEBUG,
                             DEBUG_DIR=DEBUG_DIR)
        result, switch = detect.detect_num(switchtags, im_name, switchmasks)

        # visulize
        print('count_u', count_u)
        if switch and count_u <= 1:
            for ind, res in enumerate(result):
                cv2.putText(im, 'IP: ' + res[0] + ' U: ' + res[1], (switchboxes[ind][1], switchboxes[ind][0]),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                final_result.append({'IP': res[0], 'U': [res[1]]})

        # if result and count_u < 3:
        #     switch = True
        #     for res in result:
        #         print(res[0])
        #         if len(res) >= 3:
        #             cv2.putText(im, 'IP: ' + res[1] + ' U: ' + res[2], (switchboxes[res[0]][1], switchboxes[res[0]][0]),
        #                         cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        #             final_result.append({'IP': res[1], 'U': [res[2]]})
        # else:
        #     switch = False
    return final_result, result, boxes, im, switch, server


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
            #cv2.rectangle(im_copy, (y1, x1), (y2, x2), (0, 0, 255), 3)
    #cv2.imwrite(im_name+'.jpg', im_copy)
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

# def draw_hist(myList,Title,Xlabel,Ylabel,Xmin,Xmax,Ymin,Ymax):
#     plt.hist(myList,100)
#     plt.xlabel(Xlabel)
#     plt.xlim(Xmin,Xmax)
#     plt.ylabel(Ylabel)
#     plt.ylim(Ymin,Ymax)
#     plt.title(Title)
#     plt.show()

import matplotlib.pyplot as plt
# def findjigui(im, im_name, DEBUG):
#     lower_hue_low = [20, 80, 70]
#     lower_hue_high = [31, 255, 230]
#     hsv_image = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
#     kernel_size = (10, 10)
#     mask_lower = create_hue_mask(hsv_image, lower_hue_low, lower_hue_high, kernel_size)
#     if DEBUG:
#         result_image_path = os.path.join(im_name + "_jigui.jpg")
#         cv2.imwrite(result_image_path, mask_lower)
#     jigui_list = []
#     for i in range(0, mask_lower.shape[1]):
#         col = np.array(mask_lower[:, i])
#         index = (col == 255)
#         jigui_list.append(len(col[index]))
#
#     for i in range(0, len(jigui_list)):
#         if jigui_list[i] > 10:
#             jigui_list[i] = 1
#         else:
#             jigui_list[i] = 0
#     print(jigui_list)
#     jigui = im
#     return jigui

def detecting(im_url, debug=None):
    output = sys.stdout
    outputfile = open('log.txt', 'w')
    sys.stdout = outputfile

    start = time.time()
    im_name = im_url.split('/')[-1].split('.')[0]
    im = cv2.imread(im_url)
    # if image_type == '1':
    #     #im = cv2.resize(im, (2400, 1350), interpolation=cv2.INTER_CUBIC)
    #     im = cv2.resize(im, (2740, 1540), interpolation=cv2.INTER_CUBIC)
    image_file = os.path.join(config.DETECT_IMAGE_PATH, im_name + '.jpg')
    #image_file = os.path.join('code/result', im_name + '.jpg')

    DEBUG = debug
    final_result = []

    #jigui = findjigui(im, image_file, DEBUG)

    # judge if it is switch
    final_result, result, boxes, im, switch, server = isswitch(im, im_name, DEBUG)
    print('switch: ', switch)

    if switch:
        ok = True
    else:
        print('start detect up image IP..................')
        # detect U tangs in the image
        lower_hue_low = [20, 90, 65]
        lower_hue_high = [31, 255, 255]
        ok, up_u, low_u, up_point, low_point = findUregion(im, lower_hue_low, lower_hue_high, im_name, DEBUG)
        print(up_point, low_point)
        if ok:
            for b, res in zip(boxes, result):
                if up_point <= b[0] and b[2] <= low_point:
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
                    u_list = np.arange(start_u, end_u + 1)
                    cv2.putText(im, 'IP: {}, U: {}'.format(res, u_list),
                                (boxes[ind][1], boxes[ind][0] + 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    final_result.append({'IP': res, 'U': (start_u, end_u)})

            # lower_hue_low = [23, 100, 65]
            # lower_hue_high = [31, 255, 255]
            # final_result, im = findIPregion(region, im, up_u, low_u, u_point, lower_hue_low, lower_hue_high, im_name, DEBUG)


    cv2.imwrite(image_file, im)
    end = time.time()
    print('spend time is: ', end-start)
    return ok, final_result, image_file
