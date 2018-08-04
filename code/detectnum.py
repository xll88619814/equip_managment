import cv2, os
import cPickle
import mahotas
import numpy as np
from math import *
from filters import adjustim
from imagesearch.hog import HOG
from PIL import ImageEnhance, Image
from cut_image import get_splitimages
from ImagePreprocessing import pre_proc


class detect_tags:
    def __init__(self, type_tag, ratio, thresh_w, thresh_h, thresh_gap, DEBUG, DEBUG_DIR):
        modelpath = 'code/train_nummodel/models/num_char.cpickle'
        model = open(modelpath).read()
        self.model = cPickle.loads(model)

        self.hog = HOG(orientations = 18, pixelsPerCell = (10, 10), cellsPerBlock = (1,1), transform_sqrt = True, block_norm="L2")
        self.dict = {10:'C',11:'H',12:'I',13:'S',14:'T',15:'W'}
        self.type = type_tag
        self.ratio = ratio 
        self.thresh_w = thresh_w
        self.thresh_h = thresh_h
        self.thresh_gap = thresh_gap
        self.DEBUG = DEBUG
        self.DEBUG_DIR = DEBUG_DIR
        
    def IOU(self, Reframe, GTframe):
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


    def selectcont(self, cnts, blurred):
        contours = []
    	i = 0
        width = blurred.shape[1]
    	for (c, _) in cnts:
      	    (x, y, w, h) = cv2.boundingRect(c)

            if self.thresh_w[0] <= w <= self.thresh_w[1] and self.thresh_h[0] <= h <= self.thresh_h[1] and x > 0 and h >= 0:
                if w*1.0/h < self.ratio:
                    x = x -  int(ceil((h*self.ratio-w)/2))
                    w = int(ceil(h * self.ratio))
                    #print(x,w)
                if x > 0 and y >= 0:
                    if i == 0:
                        contours.append((x, y, w, h))
                        prerect = (x, y, w, h)
                        i += 1
                    else:
                        iou =  self.IOU(prerect, (x,y,w,h))
                        if iou < 0.2:
                            contours.append((x, y, w, h))
                            prerect = (x, y, w, h)
                            i += 1
                        else:
                            if prerect[2]*prerect[3] < w*h:
                                if prerect in contours:
                                    contours.remove(prerect)
                                    i = i - 1
                                else:
                                    for k in range(count):
                                        del contours[i-1-k]
                                    i = i - count
                                contours.append((x, y, w, h))
                                prerect = (x, y, w, h) 
                                i = i + 1                  
            elif self.thresh_h[0] <= h <= self.thresh_h[1] and self.thresh_w[1] <= w <= width/2 and x > 0 and y >=0:
                #print(w, h)
                if i == 0:
                    #print('subim:', x, y, w, h)
                    blurr = blurred.copy()
                    subim = blurr[y:y+h,x:x+w]
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
                    prerect = (x, y , w, h)
                    i += len(splitimages)
                    count = len(splitimages)
                else:
                    iou =  self.IOU(prerect, (x,y,w,h))
                    if iou < 0.2:
                        blurr = blurred.copy()
                        subim = blurr[y:y+h,x:x+w]
                        #cv2.imshow('subim',subim)
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
                        prerect = (x, y , w, h)
                        i += len(splitimages)
                        count = len(splitimages) 

        clusters = []
        if not contours == []:
            clusters.append(contours[0])
            i = 0 
            for c in contours[1:]:
                if abs(c[1]+c[3]/2 - (clusters[i][1]+clusters[i][3]/2)) < 10:
                    clusters.append(c)
                    i += 1          
        return  clusters
 

    def detect_num(self, tags, image_name, masks):
        result = []
        for ind, image in enumerate(tags):
            # load the image
            width = image.shape[1] * 100 / image.shape[0]
            image = cv2.resize(image,(width, 100),interpolation=cv2.INTER_CUBIC)
            mask = cv2.resize(masks[ind],(width, 100),interpolation=cv2.INTER_CUBIC)
            #cv2.imwrite(image_name+'_'+self.type+'_'+str(ind)+'.jpg', image)
            blurred = pre_proc.proc(image, mask)

            width = blurred.shape[1] * 100 / blurred.shape[0]
            blurred = cv2.resize(blurred,(width, 100),interpolation=cv2.INTER_CUBIC)
            blurred = cv2.bitwise_not(blurred)
            edged = cv2.Canny(blurred, 50, 200)
            #cv2.imshow('edged', edged)
            #cv2.waitKey(0)
        

            _, cnts, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            cnts = sorted([(c, cv2.boundingRect(c)[0]) for c in cnts], key = lambda x: x[1])
            #cv2.drawContours(image, cnts, -1, (255, 255, 255), thickness=-1)
            #cv2.imwrite('contour.jpg',image)

            # save all contours of image
            im_copy = blurred.copy()
            for (c, _) in cnts:
                (x, y, w, h) = cv2.boundingRect(c)
                #print(x,y,w,h)
                cv2.rectangle(im_copy, (x, y), (x + w, y + h), (0, 0, 255), 1)
                #cv2.imshow('im_copy', im_copy)
                #cv2.waitKey(0)

            if self.DEBUG:
                savepath = os.path.join(self.DEBUG_DIR, image_name+'_'+self.type+'_'+str(ind)+'.jpg')
                cv2.imwrite(savepath, im_copy)


            # remain good contours
            contours = self.selectcont(cnts, blurred)
            drawim = blurred.copy()
            if self.type == 'switch':
                cluster = ''
                clusters = []
                for i, c in enumerate(contours):
                    (x, y, w, h) = c
                    #print('digit:',(x, y, w, h))
                    roi = blurred[y:y + h, x:x + w]
                    #cv2.imshow('roi', roi)
                    #cv2.waitKey(0)

                    # HOG + SVM
                    roi = cv2.resize(roi, (60,80), interpolation=cv2.INTER_CUBIC)
	            thresh = roi.copy()
	            T = mahotas.thresholding.otsu(roi)
	            thresh[thresh > T] = 255
                    thresh[thresh <= T] = 0
	            thresh = cv2.bitwise_not(thresh)
                    #cv2.imwrite('code/train_nummodel/number/'+image_name+str(ind)+'_'+str(i)+'.jpg', thresh)
                    hist = self.hog.describe(thresh)
	            digit = self.model.predict([hist])[0]
                    
                    if digit >= 10:
                        digit = self.dict[digit]
                    else:
                        digit = str(digit)
                    #print('digit:',digit)
                    #cv2.rectangle(drawim, (x, y), (x + w, y + h), (0, 0, 255), 1)
                    #cv2.putText(drawim, digit, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    if i == 0:
                        cluster += digit
                    else:
                        if x - contours[i-1][0] <= self.thresh_gap:
                            cluster += digit
                        else:
                            if cluster == 'SWTCH':
                                cluster = 'SWITCH'
                            clusters.append(cluster)
                            cluster = ''
                            cluster += digit
                #print('cluster:',cluster)
                clusters.append(cluster)
                
                #print('clusters:',clusters)
                
                #cv2.imwrite('result/'+ image_name+'_'+str(ind)+'.jpg', drawim)
            else:
                cluster = ''
                clusters = ''
                for i, c in enumerate(contours):
                    (x, y, w, h) = c
                    #print((x, y, w, h))
                    roi = blurred[y:y + h, x:x + w]
                    #cv2.imshow('roi', roi)
                    #cv2.waitKey(0)

                    # HOG + SVM
                    roi = cv2.resize(roi, (60,80), interpolation=cv2.INTER_CUBIC)
	            thresh = roi.copy()
	            T = mahotas.thresholding.otsu(roi)
	            thresh[thresh > T] = 255
                    thresh[thresh <= T] = 0
	            thresh = cv2.bitwise_not(thresh)
                    cv2.imwrite('code/train_nummodel/number/'+image_name+self.type+'_'+str(ind)+'_'+str(i)+'.jpg', thresh)
                    hist = self.hog.describe(thresh)
	            digit = self.model.predict([hist])[0]
                    #print('digit',digit)
                    #cv2.rectangle(drawim, (x, y), (x + w, y + h), (0, 0, 255), 1)
                    #cv2.putText(drawim, str(digit), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    
                    if i == 0:
                        cluster += str(digit)
                    else:
                        if x - contours[i-1][0] <= self.thresh_gap:
                            cluster += str(digit)
                        else:
                            clusters += cluster
                            clusters += '.'
                            cluster = ''
                            cluster += str(digit)
                clusters += cluster
                #print(clusters)
                #cv2.imwrite('result/'+ image_name+'_'+self.type+'_'+str(ind)+'.jpg', drawim)
            result.append(clusters)

        return result

