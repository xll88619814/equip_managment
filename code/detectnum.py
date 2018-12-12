import copy
import time
import cv2, os
import cPickle
import mahotas
from math import *
from imagesearch.hog import HOG
from cut_image import get_splitimages
from ImagePreprocessing import pre_proc


class detect_tags:
    def __init__(self, type_tag, ratio, thresh_w, thresh_h, count, DEBUG, DEBUG_DIR):
        modelpath = 'code/train_nummodel/models/num_char_new.cpickle'
        model = open(modelpath).read()
        self.model = cPickle.loads(model)

        self.hog = HOG(orientations = 18, pixelsPerCell = (10, 10), cellsPerBlock = (1, 1), transform_sqrt = True, block_norm = "L2")
        self.dict = {10:'C', 11:'E', 12:'H', 13:'R', 14:'S', 15:'T', 16:'W'}
        self.type = type_tag
        self.ratio = ratio
        self.thresh_w = thresh_w
        self.thresh_h = thresh_h
        self.count = count
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
        height, width = blurred.shape[0:2]

        for (c, _) in cnts:
            (x, y, w, h) = cv2.boundingRect(c)
            if x >= 0 and y > 0 and x+w < width and y+h <= height:
                if self.thresh_w[0] <= w <= self.thresh_w[1] and self.thresh_h[0] <= h <= self.thresh_h[1]:
                    if w*1.0/h < self.ratio:
                        if x - int(ceil((h*self.ratio-w)/2)) >= 0:
                            x = x - int(ceil((h*self.ratio-w)/2))
                            w = int(ceil(h * self.ratio))
                        #print(x,w)
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
                elif self.thresh_h[0] <= h <= self.thresh_h[1] and self.thresh_w[1] <= w <= 2*self.thresh_w[1]:#two num stick together
                    #print(w, h)
                    if i == 0:     # the first two num stick together 
                        #print('subim:', x, y, w, h)
                        blurr = blurred.copy()
                        subim = blurr[y:y+h, x:x+w]
                        #cv2.imshow('im',subim)
                        #cv2.waitKey(0)
                        splitimages = get_splitimages(subim)
                        if splitimages:
                            for j, splitimg in enumerate(splitimages):
                                h0, w0 = splitimg.shape
                                if j == 0:
                                    contours.append((x, y , w0, h0))
                                    nextx = x + w0
                                else:
                                    contours.append((nextx, y , w0, h0))
                                    nextx = nextx + w0
                            prerect = (x, y, w, h)
                            i += len(splitimages)
                            count = len(splitimages)
                    else:
                        iou = self.IOU(prerect, (x, y, w, h))
                        if iou < 0.2:
                            blurr = blurred.copy()
                            subim = blurr[y:y+h, x:x+w]
                            #print('split', x,y,w,h)
                            #cv2.imshow('subim', subim)
                            #cv2.waitKey(0)
                            splitimages = get_splitimages(subim)
                            if splitimages:
                                for j, splitimg in enumerate(splitimages):
                                    h0, w0 = splitimg.shape
                                    if j == 0:
                                        contours.append((x, y, w0, h0))
                                        nextx = x + w0
                                    else:
                                        contours.append((nextx, y , w0, h0))
                                        nextx = nextx + w0
                                prerect = (x, y, w, h)
                                i += len(splitimages)
                                count = len(splitimages)

        if not contours == []:      # Whether the all number is a row
            if not self.count == []:
                while len(contours) > self.count:
                    sumy = 0
                    for c in contours:
                        centy = c[1] + c[3]*1.0/2
                        #print('centy', centy)
                        sumy += centy
                    meany = sumy*1.0/len(contours)
                    #print('meany', meany)
                    dist = []
                    for c in contours:
                        dist.append(abs(c[1]+c[3]*1.0/2-meany))
                    ind = dist.index(max(dist))
                    del dist[ind]
                    del contours[ind]

        return contours

    def detect_num(self, tags, image_name, masks):
        result = []
        result_switch = []
        for ind, image in enumerate(tags):
            switch = False
            server = False
            # load the image
            #print(image.shape)
            width1 = image.shape[1] * 100 / image.shape[0]
            image = cv2.resize(image, (width1, 100), interpolation=cv2.INTER_CUBIC)
            mask = cv2.resize(masks[ind], (width1, 100), interpolation=cv2.INTER_CUBIC)
            #cv2.imwrite(image_name+'_'+self.type+'_'+str(ind)+'.jpg', image)
            blurred = pre_proc.proc(image, mask)

            width = blurred.shape[1] * 100 / blurred.shape[0]
            blurred = cv2.resize(blurred, (width, 100), interpolation=cv2.INTER_CUBIC)
            blurred = cv2.bitwise_not(blurred)
            edged = cv2.Canny(blurred, 50, 200)
            #cv2.imshow('edged', edged)
            #cv2.waitKey(0)

            _, cnts, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            cnts = sorted([(c, cv2.boundingRect(c)[0]) for c in cnts], key=lambda x: x[1])
            #cv2.drawContours(image, cnts, -1, (255, 255, 255), thickness=-1)
            #cv2.imwrite('contour.jpg',image)

            # save all contours of image
            im_copy = blurred.copy()
            for (c, _) in cnts:
                (x, y, w, h) = cv2.boundingRect(c)
                # print(x, y, w, h)
                cv2.rectangle(im_copy, (x, y), (x + w, y + h), (0, 0, 255), 1)
                # cv2.imshow('im_copy', im_copy)
                # cv2.waitKey(0)

            if self.DEBUG:
                savepath = os.path.join(self.DEBUG_DIR, image_name+'_'+self.type+'_'+str(ind)+'.jpg')
                cv2.imwrite(savepath, im_copy)

            # remain good contours
            contours = self.selectcont(cnts, blurred)

            cluster = ''
            clusters = []
            digits = []
            dist = []
            for i, c in enumerate(contours[1:]):
                #print(c[0] - contours[i][0])
                dist.append(c[0] - contours[i][0])
            if dist:
                max_w = max(dist)
            else:
                max_w = 0
            #print('dist..........................', dist, max_w)

            for i, c in enumerate(contours):
                (x, y, w, h) = c
                # print('digit:',(x, y, w, h))
                roi = blurred[y:y + h, x:x + w]
                # cv2.imshow('roi', roi)
                # cv2.waitKey(0)

                # HOG + SVM
                roi = cv2.resize(roi, (60, 80), interpolation=cv2.INTER_CUBIC)
                thresh = roi.copy()
                T = mahotas.thresholding.otsu(roi)
                thresh[thresh > T] = 255
                thresh[thresh <= T] = 0
                thresh = cv2.bitwise_not(thresh)
                # cv2.imwrite('code/train_nummodel/number/'+image_name+'_'+self.type+'_'+str(ind)+'_'+str(i)+'.jpg', thresh)
                hist = self.hog.describe(thresh)
                digit = self.model.predict([hist])[0]

                if digit >= 10 and self.type == 'ip':
                    digit = self.dict[digit]
                else:
                    digit = str(digit)
                digits.append(digit)
                #print('digit', digit)

                if self.type == 'u':
                    cluster += digit
                else:
                    if i == 0:
                        cluster += digit
                    else:
                        if x - contours[i-1][0] <= max_w - 25:
                            cluster += digit
                        else:
                            if 'S' in cluster and ('TC' in cluster or 'CH' in cluster or 'W' in cluster) or 'TCH' in cluster:
                                cluster = 'SWITCH'
                                switch = True
                            elif 'SE' in cluster or 'ER' in cluster or 'SR' in cluster:
                                cluster = 'SER'
                                server = True

                            clusters.append(cluster)
                            cluster = ''
                            cluster += digit
            clusters.append(cluster)
            #print(clusters)
            if not switch and not server:
                if self.type == 'ip':
                    clusters = ''
                    if dist:
                        order = copy.deepcopy(dist)
                        order.sort(reverse=True)
                        #print('dist..........................', order,dist)
                        for i, d in enumerate(digits[:-1]):
                            if order.index(dist[i]) < 3:
                                clusters = clusters + d + '.'
                            else:
                                clusters += d
                        print('ip: ', digits)
                        clusters += digits[-1]
                        #print(clusters)
                else:
                    clu = clusters[0]
                    clusters = clu
            if server and not switch:
                clu = ''
                for cl in clusters:
                    clu += cl
                clusters = clu

            print('clusters', clusters)
            if switch:
                if len(clusters) == 3:
                    if clusters[1].isdigit() and clusters[2].isdigit():
                        result_switch.append((ind, clusters))
                else:
                    if clusters[1].isdigit():
                        result_switch.append((ind, clusters))
            else:
                result.append(clusters)
        print('result: ', result)
        return result, result_switch

