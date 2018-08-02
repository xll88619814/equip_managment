import cv2
import os
import numpy as np
from pylab import * 
import mahotas
import cPickle
from PIL import ImageEnhance
from PIL import Image
from filters import adjustim
from skimage import measure
from detectnum_old import detect_num
from pyimagesearch.hog import HOG
from cut_image import get_splitimages

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
  
def IOU(Reframe,GTframe):
    x1 = Reframe[0]
    y1 = Reframe[1]
    width1 = Reframe[2]
    height1 = Reframe[3]

    x2 = GTframe[0]
    y2 = GTframe[1]
    width2 = GTframe[2]
    height2 = GTframe[3]

    endx = max(x1+width1,x2+width2)
    startx = min(x1,x2)
    width = width1+width2-(endx-startx)

    endy = max(y1+height1,y2+height2)
    starty = min(y1,y2)
    height = height1+height2-(endy-starty)

    if width <=0 or height <= 0:
        ratio = 0
    else:
        Area = width*height
        Area1 = width1*height1
        Area2 = width2*height2
        ratio = Area*1.0/(Area1+Area2-Area)
    return ratio


def detect_redlight(im, debug_dir, im_name):
    # Find red regions
    red_hue_low = [150, 178, 100]
    red_hue_high = [180, 255, 200]
    hsv_image = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    # Get lower hue
    kernel_size = (5,5)
    mask_lower= create_hue_mask(hsv_image, red_hue_low, red_hue_high, kernel_size)
    if debug_dir:
        result_image_path = os.path.join(debug_dir, im_name + "_redlight.jpg")
        cv2.imwrite(result_image_path, mask_lower)
    # Find suitable equipment labels
    labels = measure.label(mask_lower, connectivity=2)
    pro = measure.regionprops(labels)
    ok = True
    if len(pro) > 0:
        ok = False
        circle_color = (0, 255, 255)
        for p in pro:
            (x, y) = p.centroid
            radius = x - p.bbox[0] + 7
            #print(y,x, radius)
            cv2.circle(im, (int(y), int(x)), int(radius), circle_color, 5)
    return ok, len(pro), im

def selectcont(cnts, blurred):
    '''
    im_copy = blurred.copy()
    for (c, _) in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        print(x,y,w,h)
        cv2.rectangle(im_copy, (x, y), (x + w, y + h), (0, 0, 255), 1)
        cv2.imshow('im_copy', im_copy)
        cv2.waitKey(0)
     '''

    contours = []
    i = 0
    for (c, _) in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        if w >= 7 and w <= 45 and h >=50 and h <= 75 and x > 0 and h > 0:
            if w*1.0/h < 0.5:
                x = x -  int(ceil((h*0.5-w)/2))
                w = int(ceil(h * 0.5))
            if w*1.0/h > 0.6:
                y = y -  int(ceil((w/0.6-h)/2))
                h = int(ceil(w/0.6))
            if x > 0 and y > 0:
                if i == 0:
                    prerect = (x, y, w, h)
                    contours.append(prerect)
                    i += 1
                else:
                    iou =  IOU(prerect, (x,y,w,h))
                    if iou < 0.2:
                        prerect = (x, y, w, h)
                        contours.append(prerect)
                        i += 1
        elif h >=45 and h <= 65 and w > 45 and x >0 and y >0:
            #print(w)
            if i == 0:
                subim = blurred[y:y+h,x:x+w]
                #cv2.imshow('im',subim)
                #cv2.waitKey(0)
                splitimages = get_splitimages(subim)
                for j, splitimg in enumerate(splitimages):
                    h0, w0 = splitimg.shape
                    if j == 0:
                        contours.append((x, y , w0, h0))
                        nextx = x + w0
                    else:
                        contours.append((nextx, y , w0, h0))
                        nextx = nextx + w0
            else:
                iou =  IOU(prerect, (x,y,w,h))
                if iou < 0.2:
                    subim = blurred[y:y+h,x:x+w]
                    #cv2.imshow('im',subim)
                    #cv2.waitKey(0)
                    splitimages = get_splitimages(subim)
                    for j, splitimg in enumerate(splitimages):
                        h0, w0 = splitimg.shape
                        if j == 0:
                            contours.append((x, y , w0, h0))
                            nextx = x + w0
                        else:
                            contours.append((nextx, y , w0, h0))
                            nextx = nextx + w0
                #print(len(splitimages))
            prerect = (x, y, w, h)
            i += len(splitimages)

    return  contours

def unum(region, im_name, ind):
    model_path = 'train_nummodel/models/newmysvm_716.cpickle'
    model = open(model_path).read()
    model = cPickle.loads(model)

    # initialize the HOG descriptor
    hog = HOG(orientations = 18, pixelsPerCell = (10, 10), cellsPerBlock = (1,1), transform_sqrt = True, block_norm="L2")

    image = cv2.resize(region,(120, 100),interpolation=cv2.INTER_CUBIC)
    image = Image.fromarray(image)
    enh_con = ImageEnhance.Contrast(image)
    contrast = 8
    im_con = enh_con.enhance(contrast)
    im_con = np.asarray(im_con)
    im_con = adjustim('curve4.acv', im_con)
    #cv2.imshow('edged:', im_con)
    #cv2.waitKey(0)
    gray = cv2.cvtColor(im_con, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    edged = cv2.Canny(blurred, 10, 100)
    #cv2.imshow('edged:', edged)
    #cv2.waitKey(0)
    _, cnts, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted([(c, cv2.boundingRect(c)[0]) for c in cnts], key = lambda x: x[1])
    contours = selectcont(cnts, blurred)

    u_num = ''
    for i, c in enumerate(contours):
        (x, y, w, h) = c
        roi = blurred[y:y + h, x:x + w]
        #print(x,y)
        # HOG + SVM
        roi = cv2.resize(roi,(60,80),interpolation=cv2.INTER_CUBIC)
        cv2.imshow('roi', roi)
        cv2.waitKey(0)
	thresh = roi.copy()
	T = mahotas.thresholding.otsu(roi)
	thresh[thresh > T] = 255
        thresh[thresh <= T] = 0
	thresh = cv2.bitwise_not(thresh)
        #cv2.imwrite('train_nummodel/number/'+im_name+'_'+str(ind)+'_'+str(i)+'.jpg', thresh)
        hist = hog.describe(thresh)
	digit = model.predict([hist])[0]
        u_num += str(digit)

    return u_num	
    

def findregion(im, im_name, debug_dir):

    # Find jigui tags
    lower_hue_low = [25, 127, 90]
    lower_hue_high = [32, 255, 230]
    hsv_image = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    kernel_size = (5, 5)
    mask_lower= create_hue_mask(hsv_image, lower_hue_low, lower_hue_high, kernel_size)
    if debug_dir:
        result_image_path = os.path.join(debug_dir, im_name + "_jigui.jpg")
        cv2.imwrite(result_image_path, mask_lower)
    labels = measure.label(mask_lower, connectivity=2)
    pro = measure.regionprops(labels)

    # remain good regions and detect u num
    im_copy = im.copy()
    boxes = []
    u_num = []
    for ind, p in enumerate(pro): 
        (x1, y1, x2, y2) = p.bbox
        if 10 < y2-y1 < 95 and 10 < x2-x1 < 70:
            # detect u num
            uregion = im[x1:x2, y1:y2, :]
            #cv2.imwrite(im_name+'_'+str(ind)+'.jpg', uregion)
            #cv2.imshow('u',uregion)
            #cv2.waitKey(0)
            num = unum(uregion, im_name, ind)
            print('detect:', num)
            u_num.append(int(num))
            boxes.append(p.bbox)
            cv2.rectangle(im_copy, (y1, x1), (y2, x2), (0, 0, 255), 3)           
            #print(y2-y1, x2-x1)
    cv2.imwrite('im.jpg', im_copy)

    ok = True
    up_u = 0
    low_u = 0
    region = im.copy()
    print('jigui tags is: ', len(boxes))
    # adjust jigui redion
    if len(boxes) == 4:
        if u_num[0] == u_num[1] and u_num[2] == u_num[3]:
            low_u = u_num[2]
            up_u = u_num[0]
         
            boxes.sort(key=lambda x:x[0])
            up_region = boxes[:2]
            low_region = boxes[2:]    
            up_region.sort(key=lambda x:x[1])    # up_region[0] is left_up,  up_region[1] is right_up
            low_region.sort(key=lambda x:x[1])    # low_region[0] is left_low, low_region[1] is right_low
            src_point=np.float32([[up_region[0][1],up_region[0][2]-15],[low_region[0][1],low_region[0][2]],
                             [low_region[1][3],low_region[1][2]],[up_region[1][3],up_region[1][2]-15]])
            #print(src_point)
            dsize=(1000, 800)
            dst_point = np.float32([[0,0],[0,dsize[1]-1],[dsize[0]-1,dsize[1]-1],[dsize[0]-1,0]])
            h, s = cv2.findHomography(src_point, dst_point, cv2.RANSAC, 10)   # projection transformation
            region = cv2.warpPerspective(im, h, dsize)
            if debug_dir:
                result_image_path = os.path.join(debug_dir, im_name + "_jigui.jpg")
                cv2.imwrite(result_image_path, region)
        else:
            ok = False
            print('detect u incorrectly')      
    else:
        ok = False
        print('count of jigui tags is not four')

    return ok, region, up_u, low_u, up_region[0][2]

def isswitch(im, im_name, debug_dir):
    lower_hue_low = [100, 76, 50]
    lower_hue_high = [112, 204, 130]
    hsv_image = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    kernel_size = None
    mask_lower= create_hue_mask(hsv_image, lower_hue_low, lower_hue_high, kernel_size)
    if debug_dir:
        result_image_path = os.path.join(debug_dir, im_name + "_switch.jpg")
        cv2.imwrite(result_image_path, mask_lower)
    labels = measure.label(mask_lower, connectivity=2)
    pro = measure.regionprops(labels)
    switchtags = []
    for p in pro:
        (x1, y1, x2, y2) = p.bbox
        if (x2-x1) > 15 and (y2-y1) > 90:
            print(p.bbox)
            switchtags.append(im[x1+2:x2-2,y1+2:y2-2,:])
    switch = False
    if len(switchtags) > 0:
        #print('box:', len(switchbox))
        switch = True
        
    return switch, switchtags


def gettag(im, debug_dir, im_name):
    # judge if it is switch
    
    switch, switchtags = isswitch(im, im_name, debug_dir)
    print('switch: ', switch)
    if switch:
        ok = False
        tag = switchtags
        points = []
        u = []
        u_point = 0
        for ind, tag in enumerate(switchtags):
            cv2.imwrite(im_name + '_' +str(ind) +'.jpg', tag)
    else:
        # get jigui region and max u and min u
        ok, im, up_u, low_u, u_point = findregion(im, im_name, debug_dir)
        tag = []
        points = []
        u = []
        if ok == True:
            sum_u = up_u - low_u
            height_u = im.shape[0]*1.0/sum_u
            #print('sum_u:', sum_u, 'height_u: ',height_u)
            
            # Find yellow regions
            #lower_hue_low = [25, 153, 153]
            #lower_hue_high = [29, 255, 230]
            lower_hue_low = [25, 127, 80]
            lower_hue_high = [31, 255, 230]
            hsv_image = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

            kernel_size = (10,10)
            mask_lower= create_hue_mask(hsv_image, lower_hue_low, lower_hue_high, kernel_size)
            if debug_dir:
                result_image_path = os.path.join(debug_dir, im_name + "_equipment.jpg")
                cv2.imwrite(result_image_path, mask_lower)

            labels = measure.label(mask_lower, connectivity=2)
            pro = measure.regionprops(labels)
            # remain good yellow regions
            boxes = []
            imm = im.copy()
            min_width = 1000
            for p in pro:
                (x1, y1, x2, y2) = p.bbox
                width = y2 - y1
                if width > 100:
                    if min_width > width:
                        min_width = width
            #print('min_width:', min_width)
            box = []
            for p in pro:
                (x1, y1, x2, y2) = p.bbox
                if min_width-5 < y2-y1 < min_width + 15:
                    box.append(p.bbox)
            for i, b in enumerate(box):
                if i%2 == 0:
                    #(x1,y1,x2,y2) = pro[i].bbox   
                    #cv2.rectangle(imm, (y1, x1), (y2, x2), (0, 0, 255), 3)
                    subim = im[b[0]+2:b[2]-2,b[1]+2:b[3]-2,:]
                    cv2.imwrite(im_name+'_'+str(i)+'_'+'tag.jpg', subim)
                    tag.append(subim)
                    points.append((b[0], b[1]))
                
                    # compute u
                    end_u = int(round(up_u - b[0]/height_u) -1)
                    start_u = int(round(up_u - box[i+1][2]/height_u))
                    u.append((start_u, end_u))
                    #print(end_u, start_u)
                    print('equipment '+str(i/2)+': ', np.arange(start_u, end_u+1))      
                
    return ok, tag, points, u, u_point

def detecting(im, debug_dir, im_name):
    '''
    # detect red light
    light, count, img = detect_redlight(im, 'debug/', im_name)
    if not light:
        print('image has {} red lights'.format(count))
    '''
    ok, tags, points, u, u_point = gettag(im, debug_dir, im_name)
    #print(ok, len(tags), len(points), len(u))
    if not ok:
        return False

    IP = detect_num(im, tags, im_name, debug_dir)
    for ind, ip in enumerate(IP):
        u_list = np.arange(u[ind][0], u[ind][1]+1)
        cv2.putText(im, "ip: {}, u: {}".format(ip, u_list), (points[ind][1]+400,points[ind][0]+u_point+100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    savepath = 'result/'+im_name+'.jpg'
    cv2.imwrite(savepath, im)
    return ok
