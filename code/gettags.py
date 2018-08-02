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


def findregion(im, im_name, DEBUG):

    # Find jigui tags
    lower_hue_low = [23, 150, 50]
    lower_hue_high = [29, 255, 230]
    hsv_image = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    kernel_size = (9, 9)
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
        if 35 < y2-y1 < 80 and 15 < x2-x1 < 40 and y2<0.9*im_width and y1>0.1*im_width and x1>0.1*im_height:
            i += 1
            utags.append(im[x1-5:x2+5,y1-5:y2+5,:])
            uboxes.append(p.bbox)
            umasks.append(mask_lower[x1-5:x2+5,y1-5:y2+5])
            cv2.rectangle(im_copy, (y1, x1), (y2, x2), (0, 0, 255), 3)               
            #print(y2-y1, x2-x1)
            #subim = im[x1-5:x2+5,y1-5:y2+5,:]
            #submask = mask_lower[x1-5:x2+5,y1-5:y2+5]
            #width = submask.shape[1] * 100 / submask.shape[0]
            #subim = cv2.resize(subim,(width, 100),interpolation=cv2.INTER_CUBIC)
            #submask = cv2.resize(submask,(width, 100),interpolation=cv2.INTER_CUBIC)
            #cv2.imwrite('image/'+im_name+'_'+str(i)+'_'+'u.jpg', subim)
            #cv2.imwrite('image/'+im_name+'_'+str(i)+'_'+'u_mask.jpg', submask)
    cv2.imwrite('im.jpg', im_copy)

    if len(uboxes) == 4 or len(uboxes) == 2:
        detect = detect_tags(type_tag = 'u', ratio = 0.6, thresh_w = [17, 45], thresh_h = [42, 60], DEBUG=DEBUG, DEBUG_DIR=DEBUG_DIR)
        u_num = detect.detect_num(utags, im_name, umasks)
    #print(u_num)
    for u in u_num:
        if len(u) > 2:
            u_num.remove(u)
            u_num.append(u[:2])
        elif len(u) < 2:
            u_num.remove(u)
    set_unum = list(set(u_num))
    for s in set_unum:
        if not s:
            set_unum.remove(s)
    set_unum.sort()
    print('u_num:', set_unum)

    ok = True
    up_u = 0
    low_u = 0
    region = im.copy()
    print('jigui tags is: ', len(uboxes))
    # adjust jigui redion
    if len(set_unum) == 2 and len(uboxes) == 4:

        low_u = int(set_unum[0])
        up_u = int(set_unum[1])
        #print('low_u:', low_u, 'up_u:', up_u)
         
        uboxes.sort(key=lambda x:x[0])
        up_region = uboxes[:2]
        low_region = uboxes[2:]    
        up_region.sort(key=lambda x:x[1])    # up_region[0] is left_up,  up_region[1] is right_up
        low_region.sort(key=lambda x:x[1])    # low_region[0] is left_low, low_region[1] is right_low
        src_point=np.float32([[up_region[0][1],up_region[0][2]-15],[low_region[0][1],low_region[0][2]],
                             [low_region[1][3],low_region[1][2]],[up_region[1][3],up_region[1][2]-15]])
        u_point = up_region[0][2]
        dsize=(1000, 800)
        dst_point = np.float32([[0,0],[0,dsize[1]-1],[dsize[0]-1,dsize[1]-1],[dsize[0]-1,0]])
        h, s = cv2.findHomography(src_point, dst_point, cv2.RANSAC, 10)   # projection transformation
        region = cv2.warpPerspective(im, h, dsize)
        if DEBUG:
            result_image_path = os.path.join(DEBUG_DIR, im_name + "_jigui.jpg")
            cv2.imwrite(result_image_path, region)
    elif len(set_unum) == 2 and len(uboxes) == 2:
        print('u tag is two....................')
        low_u = int(set_unum[0])
        up_u = int(set_unum[1])
        region = im[uboxes[0][0]:uboxes[1][2],:,:]
        width = int(region.shape[1] * 800 / region.shape[0]*1.05)
        region = cv2.resize(region,(width, 800),interpolation=cv2.INTER_CUBIC)
        u_point = uboxes[0][0]
        if DEBUG:
            result_image_path = os.path.join(DEBUG_DIR, im_name + "_jigui.jpg")
            cv2.imwrite(result_image_path, region)
    else:
        ok = False
        u_point = 0
        print('detect u incorrectly')      
   
    return ok, region, up_u, low_u, u_point


def isswitch(im, im_name):
    #lower_hue_low = [25, 127, 230]
    #lower_hue_high = [30, 255, 255]
    lower_hue_low = [23, 170, 80]
    lower_hue_high = [30, 255, 255]
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
        if (x2-x1) > 15 and (y2-y1) > 180:
            #i += 1
            switchboxes.append(p.bbox)
            switchtags.append(im[x1-5:x2+5,y1-5:y2+5,:])
            switchmasks.append(mask_lower[x1-5:x2+5,y1-5:y2+5])
            #subim = im[x1-5:x2+5,y1-5:y2+5,:]
            #submask = mask_lower[x1-5:x2+5,y1-5:y2+5]
            #width = submask.shape[1] * 100 / submask.shape[0]
            #subim = cv2.resize(subim,(width, 100),interpolation=cv2.INTER_CUBIC)
            #submask = cv2.resize(submask,(width, 100),interpolation=cv2.INTER_CUBIC)
            #cv2.imwrite('image/'+im_name+'_'+str(i)+'_'+'switch.jpg', subim)
            #cv2.imwrite('image/'+im_name+'_'+str(i)+'_'+'switch_mask.jpg', submask)
    switch = False
    if len(switchboxes) > 0:
        #print('switchboxes:', len(switchboxes))
        switch = True
        
    return switch, switchboxes, switchtags, switchmasks


def detecting(im_url, debug=False):
    im = cv2.imread(im_url)
    im_name = im_url.split('/')[-1].split('.')[0]
    DEBUG = debug
    image = im.copy()
    final_result = []

    # judge if it is switch
    switch, switchboxes, switchtags, switchmasks = isswitch(im, im_name)
    print('switch: ', switch)
    if switch:
        ok = True
        detect = detect_tags(type_tag = 'switch', ratio = 0.6, thresh_w = [20, 55], thresh_h = [40, 65], DEBUG=DEBUG, DEBUG_DIR=DEBUG_DIR)
        result = detect.detect_num(switchtags, im_name, switchmasks)
        u = []
        # visulize
        #for ind, res in enumerate(result):
        #    point = switchboxes[ind][0:2]
        #    u.append(res[2:])
        #    cv2.putText(image, "{}, IP: {}, U: {}".format(res[0], res[1], res[2:]), (point[1],point[0]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        #savepath = 'result/'+im_name+'.jpg'
        #cv2.imwrite(savepath, image)
    else:
        # get jigui region and max u and min u
        ok, img, up_u, low_u, u_point = findregion(im, im_name, DEBUG)

        if ok:
            print('start detect IP..................')
            sum_u = up_u - low_u
            height_u = img.shape[0]*1.0/sum_u
            #print('sum_u:', sum_u, 'height_u: ',height_u)
            
            # Find yellow regions
            lower_hue_low = [25, 127, 80]
            lower_hue_high = [31, 255, 230]
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
                if 160 > width > 100:
                    if min_width > width:
                        min_width = width
            #print('min_width:', min_width)
            box = []
            for p in pro:
                (x1, y1, x2, y2) = p.bbox
                if min_width <= y2-y1 < min_width + 15:
                    box.append(p.bbox)
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
                        #width = submask.shape[1] * 100 / submask.shape[0]
                        #subim = cv2.resize(subim,(width, 100),interpolation=cv2.INTER_CUBIC)
                        #submask = cv2.resize(submask,(width, 100),interpolation=cv2.INTER_CUBIC)
                        #cv2.imwrite('image/'+im_name+'_'+str(i)+'_'+'tag.jpg', subim)
                        #cv2.imwrite('image/'+im_name+'_'+str(i)+'_'+'tag_mask.jpg', submask)
                        if DEBUG:
                            result_image_path = os.path.join(DEBUG_DIR, im_name+'_'+str(i)+'_'+'tag.jpg')
                            cv2.imwrite(result_image_path, subim)
                        iptags.append(subim)
                        ipmasks.append(submask)
                        ipboxes.append(b)
                
                        # compute u
                        end_u = int(round(up_u - b[0]/height_u) -1)
                        start_u = int(round(up_u - box[i+1][2]/height_u))
                        u.append((start_u, end_u))
                        #print(end_u, start_u)
                        #print('equipment '+str(i/2)+': ', np.arange(start_u, end_u+1))

                detect = detect_tags(type_tag = 'ip', ratio = 0.6, thresh_w = [18, 40], thresh_h = [46, 60], DEBUG=DEBUG, DEBUG_DIR=DEBUG_DIR)
                result = detect.detect_num(iptags, im_name, ipmasks)
                # visulize
                #for ind, res in enumerate(result):
                #    u_list = np.arange(u[ind][0], u[ind][1]+1)
                #    point = ipboxes[ind][0:2]
                #    cv2.putText(image, "IP: {}, U: {}".format(res, u_list), (image.shape[1]/3,point[0]+u_point+80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                #savepath = 'result/'+im_name+'.jpg'
                #cv2.imwrite(savepath, image)
                final_result = []
                for i, res in enumerate(result):
                    final_result.append({'ID':res, 'U':u[i]})
                print('final result: ', final_result)
            #else:
                #final_result = []
                #u = []
                #savepath = 'result/'+im_name+'.jpg'
                #cv2.imwrite(savepath, image)
        #else:
            #final_result = []
            #u = []
            #savepath = 'result/'+im_name+'.jpg'
            #cv2.imwrite(savepath, image)
    

    return ok, final_result

