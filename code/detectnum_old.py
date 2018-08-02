from pyimagesearch.hog import HOG
import cPickle
import mahotas
import cv2
from math import *
import os
from PIL import ImageEnhance
from PIL import Image
import numpy as np
from cut_image import get_splitimages
from filters import adjustim


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

def selectcont(cnts, blurred):
    contours = []
    i = 0
    for (c, _) in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        if w >= 8 and w <= 35 and h >=50 and h <= 75 and x > 0 and h > 0:
            if w*1.0/h < 0.35:
                x = x -  int(ceil((h*0.35-w)/2))
                w = int(ceil(h * 0.35))
                    #print(x,w)
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
                else:
                    if prerect[2]*prerect[3] < w*h:
                        contours.remove(prerect)
                        prerect = (x, y, w, h)
                        contours.append(prerect)
                        i += 1
        elif h >=50 and h <= 75 and w > 35:
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

'''
def detect_num_old(im, tags, image_name, points):
    # load the model
    model_path = 'Number_Detection/models/newmysvm_716.cpickle'
    model = open(model_path).read()
    model = cPickle.loads(model)
    #print model

    # initialize the HOG descriptor
    hog = HOG(orientations = 18, pixelsPerCell = (10, 10), cellsPerBlock = (1,1), transform_sqrt = True, block_norm="L2")

    for ind, image in enumerate(tags):
        # load the image
        image = cv2.resize(image,(600, 100),interpolation=cv2.INTER_CUBIC)
        cv2.imwrite('mask_'+str(ind)+'.jpg', image)

      
        image = Image.fromarray(image)
        enh_con = ImageEnhance.Contrast(image)
        contrast = 15
        im_con = enh_con.enhance(contrast)
        im_con = np.asarray(im_con)
 
        gray = cv2.cvtColor(im_con, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 0)


        im_con = adjustim(image)
        
        gray = cv2.cvtColor(im_con, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        cv2.imwrite('con_'+str(ind)+'.jpg', blurred)
        edged = cv2.Canny(blurred, 10, 100)
        #cv2.imshow('edged', edged)
        #cv2.waitKey(0)


        _, cnts, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted([(c, cv2.boundingRect(c)[0]) for c in cnts], key = lambda x: x[1])
        #cv2.drawContours(image, cnts, -1, (255, 255, 255), thickness=-1)
        #cv2.imwrite('contour.jpg',image)


        # save all contours of image
        contours = []
        im_copy = im_con.copy()
        for (c, _) in cnts:
            (x, y, w, h) = cv2.boundingRect(c)
            #print(x,y,w,h)
            cv2.rectangle(im_copy, (x, y), (x + w, y + h), (0, 0, 255), 1)
            #cv2.imshow('im_copy', im_copy)
            #cv2.waitKey(0)
        savepath = 'debug/'+image_name+'_'+str(ind)+'.jpg'
        cv2.imwrite(savepath, im_copy)


        # remain good contours
        contours = selectcont(cnts, gray)
        #print(len(contours))

        im_copy = im_con.copy()
        for c in contours:
            (x, y, w, h) = c
            #print(x,y,w,h)
            cv2.rectangle(im_copy, (x, y), (x + w, y + h), (0, 0, 255), 1)
            #cv2.imshow('im_copy', im_copy)
            #cv2.waitKey(0)

        clusters = []
        cluster = []
        ip = []
        for i, c in enumerate(contours):
            (x, y, w, h) = c
            roi = gray[y - 1:y + h + 1, x - 1:x + w + 1]

            # HOG + SVM
            roi = cv2.resize(roi,(60,80),interpolation=cv2.INTER_CUBIC)
	    thresh = roi.copy()
	    T = mahotas.thresholding.otsu(roi)
	    thresh[thresh > T] = 255
            thresh[thresh <= T] = 0
	    thresh = cv2.bitwise_not(thresh)
            cv2.imwrite('Number_Detection/number/'+image_name+'_'+str(ind)+'_'+str(i)+'.jpg', thresh)
            hist = hog.describe(thresh)
	    digit = model.predict([hist])[0]
            
            if i == 0:
                ip.append(digit)
            else:
                if x - contours[i-1][0] <= 35:
                    ip.append(digit)
                elif 35 < x - contours[i-1][0] < 50:
                    cluster.append(ip)
                    ip = []
                    ip.append(digit)
                else:
                    cluster.append(ip)
                    ip = []
                    clusters.append(cluster)
                    cluster = []
                    ip.append(digit)
      
        
            #cv2.rectangle(im_con, (x, y), (x + w, y + h), (0, 0, 255), 1)
            #cv2.putText(im_con, str(digit), (x + 5, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cluster.append(ip)
        clusters.append(cluster)

        ip = ''
        for ip_num in clusters[0]:
            for n in ip_num:
                ip += str(n)
            ip += '.'

        u = ''
        for u_num in clusters[1:]:
            for n in u_num:
                for nn in n:
                    u += str(nn)
            u += ','
        
        print "ip: {}, u: {}".format(ip[:-1], u[:-1])
        #print(points[ind])
        cv2.putText(im, "ip: {}, u: {}".format(ip[:-1], u[:-1]), (points[ind][1],points[ind][0]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    savepath = 'result/'+image_name+'.jpg'
    cv2.imwrite(savepath, im)

    return True
'''

def detect_num(im, tags, image_name, debug_dir):
    # load the model
    model_path = 'train_nummodel/models/newmysvm_716.cpickle'
    model = open(model_path).read()
    model = cPickle.loads(model)

    # initialize the HOG descriptor
    hog = HOG(orientations = 18, pixelsPerCell = (10, 10), cellsPerBlock = (1,1), transform_sqrt = True, block_norm="L2")

    ip = []
    for ind, image in enumerate(tags):
        # load the image
        image = cv2.resize(image,(400, 100),interpolation=cv2.INTER_CUBIC)

        image = Image.fromarray(image)
        enh_con = ImageEnhance.Contrast(image)
        contrast = 8
        im_con = enh_con.enhance(contrast)
        im_con = np.asarray(im_con)
        
        im_con = adjustim('curve/curve4.acv', im_con)
        #cv2.imshow('im_con', im_con)
        #cv2.waitKey(0)

        gray = cv2.cvtColor(im_con, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 0)
        cv2.imwrite(debug_dir+'con_'+str(ind)+'.jpg', blurred)
        edged = cv2.Canny(blurred, 10, 100)
        #cv2.imshow('edged', edged)
        #cv2.waitKey(0)


        _, cnts, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted([(c, cv2.boundingRect(c)[0]) for c in cnts], key = lambda x: x[1])
        #cv2.drawContours(image, cnts, -1, (255, 255, 255), thickness=-1)
        #cv2.imwrite('contour.jpg',image)

        #'''
        # save all contours of image
        im_copy = im_con.copy()
        for (c, _) in cnts:
            (x, y, w, h) = cv2.boundingRect(c)
            #print(x,y,w,h)
            cv2.rectangle(im_copy, (x, y), (x + w, y + h), (0, 0, 255), 1)
            #cv2.imshow('im_copy', im_copy)
            #cv2.waitKey(0)
        savepath = debug_dir+image_name+'_'+str(ind)+'.jpg'
        cv2.imwrite(savepath, im_copy)
        #'''

        # remain good contours
        contours = selectcont(cnts, blurred)
        #print(len(contours))
        '''
        im_copy = im_con.copy()
        for c in contours:
            (x, y, w, h) = c
            #print(x,y,w,h)
            cv2.rectangle(im_copy, (x, y), (x + w, y + h), (0, 0, 255), 1)
            cv2.imshow('im_copy', im_copy)
            cv2.waitKey(0)
        '''

        num = ''
        cluster = ''
        for i, c in enumerate(contours):
            (x, y, w, h) = c
            roi = gray[y:y + h, x:x + w]
            #cv2.imshow('roi', roi)
            #cv2.waitKey(0)

            # HOG + SVM
            roi = cv2.resize(roi, (60,80), interpolation=cv2.INTER_CUBIC)
	    thresh = roi.copy()
	    T = mahotas.thresholding.otsu(roi)
	    thresh[thresh > T] = 255
            thresh[thresh <= T] = 0
	    thresh = cv2.bitwise_not(thresh)
            #cv2.imwrite('train_nummodel/number/'+image_name+'_'+str(ind)+'_'+str(i)+'.jpg', thresh)
            hist = hog.describe(thresh)
	    digit = model.predict([hist])[0]
            
            if i == 0:
                num += str(digit)
            else:
                if x - contours[i-1][0] <= 35:
                    num += str(digit)
                else:
                    cluster += num
                    cluster += '.'
                    num = ''
                    num += str(digit)
      
        cluster += num
        print "ip: {}".format(cluster)
        ip.append(cluster)
    print(ip)

    return ip


