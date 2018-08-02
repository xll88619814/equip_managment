#############################################
# Universidad Tecnica Particular de Loja    #
#############################################
# Professor:                                #
# Rodrigo Barba        lrbarba@utpl.edu.ec  #
#############################################
# Students:                                 #
# Marcelo Bravo        mdbravo4@utpl.edu.ec #
# Galo Celly           gscelly@utpl.edu.ec  #
# Nicholas Earley      nearley@utpl.edu.ec  #
#############################################

# python detect_numbers_images.py --model models/svm.cpickle --image images/f5a.jpg

# import the necessary packages
from pyimagesearch.hog import HOG
from pyimagesearch import dataset
import argparse
import cPickle
import mahotas
import cv2
#import os
#uncomment for use audio
import mask as msk
import numpy as np
import matplotlib.pyplot as plt

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required = True,
	help = "path to where the model will be stored")
ap.add_argument("-i", "--image", required = True,
	help = "path to the image file")
args = vars(ap.parse_args())
numeros=[]

# load the model
model = open(args["model"]).read()
model = cPickle.loads(model)
print model

# initialize the HOG descriptor
hog = HOG(orientations = 18, pixelsPerCell = (10, 10),
	cellsPerBlock = (1,1), transform_sqrt = True)


# load the image
image = cv2.imread(args["image"])

# apply filters (Added, subtracted, bitwise Or)
M = np.ones(image.shape, dtype = "uint8") * 5
added = cv2.add(image, M)
#cv2.imshow('add', added)
#cv2.waitKey(0)
M = np.ones(image.shape, dtype = "uint8") * 20
subtracted = cv2.subtract(image, M)
#cv2.imshow("Subtracted", subtracted)
#cv2.waitKey(0)
gray = cv2.cvtColor(subtracted, cv2.COLOR_BGR2GRAY)
bitwise_Or = cv2.bitwise_not(added,subtracted)
#cv2.imshow("not", bitwise_Or)
#cv2.waitKey(0)

# Convert image to gray scale
gray=cv2.cvtColor(bitwise_Or, cv2.COLOR_BGR2GRAY)
gray = cv2.cvtColor(subtracted, cv2.COLOR_BGR2GRAY)

# location you recognize (using a mask)
mask = np.zeros(gray.shape[:2], dtype = "uint8")
(cX, cY) = (gray.shape[1] / 2, gray.shape[0] / 2)
cv2.rectangle(mask, (cX - 260, cY - 310), (cX + 170 , cY - 215), 255, -1) # change this line for other image (first mask)
#cv2.imshow('mask',mask)
#cv2.waitKey(0)
masked = cv2.bitwise_and(gray, gray, mask = mask)
#cv2.imshow("Mask Applied to Image", masked)
#cv2.waitKey(0)


# blur the image, find edges, and then find contours along
# the edged regions
blurred = cv2.GaussianBlur(masked, (5, 5), 0)
edged = cv2.Canny(blurred, 30, 150)
#cv2.imshow('edged', edged)
#cv2.waitKey(0)
_, cnts, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#cv2.waitKey(0)
# sort the contours by their x-axis position, ensuring
# that we read the numbers from left to right
cnts = sorted([(c, cv2.boundingRect(c)[0]) for c in cnts], key = lambda x: x[1])
#cv2.drawContours(image, cnts, -1, (255, 255, 255), thickness=-1) 
#cv2.imwrite('contour.jpg',image)

contours = []
prec, _ = cnts[2]
(prex, prey, prew, preh) = cv2.boundingRect(prec)
#cv2.rectangle(image, (prex, prey), (prex + prew, prey + preh), (0, 0, 255), 1)
#cv2.imshow('image',image)
#cv2.waitKey(0)
contours.append((prex, prey, prew, preh))
for (c, _) in cnts[3:]:
#	# compute the bounding box for the rectangle
	(x, y, w, h) = cv2.boundingRect(c)
        if x>=prex and y>=prey and (x+w)<=(prex+prew) and (y+h)<=(prey+preh):
                continue 
        else:
                #cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 1)
                #cv2.imshow('image',image)
                #cv2.waitKey(0)
                (prex, prey, prew, preh) = (x,y,w,h)
                contours.append((x,y,w,h))
       

# loop over the contours
for c in contours:
	# compute the bounding box for the rectangle
	(x, y, w, h) = c
        #print(w,h)
	# if the width is at least 7 pixels and the height
	# is at least 20 pixels, the contour is likely a digit
	if w >= 7 and h >= 28:
		# crop the ROI and then threshold the grayscale
		# ROI to reveal the digit
        # apply filer
		roi = gray[y:y + h, x:x + w]
                #cv2.imshow('roi',roi)
                #cv2.waitKey(0)
		thresh = roi.copy()
		T = mahotas.thresholding.otsu(roi)
		thresh[thresh > T] = 255
		thresh = cv2.bitwise_not(thresh)
                #cv2.imshow('otsu',thresh)
                #cv2.waitKey(0)
      
		# deskew the image center its extent
		thresh = dataset.deskew(thresh, 20)
		thresh = dataset.center_extent(thresh, (20, 20))
		cv2.imshow("thresh", thresh)
                cv2.waitKey(0)

		# extract features from the image and classify it
		hist = hog.describe(thresh)
                print(hist.shape)
		digit = model.predict([hist])[0]
		print "Creo que el numero es: %d" % (digit)
    
		# draw a rectangle around the digit, the show what the
		# digit was classified as
		cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 1)
		cv2.putText(image, str(digit), (x - 5, y - 5),
			cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
		cv2.imshow("image", image)
        # prees a key to continue
		cv2.waitKey(0)

# load the image and apply filters (subtracted)
image = cv2.imread(args["image"])
M = np.ones(image.shape, dtype = "uint8") * 10
subtracted = cv2.subtract(image, M)

# Convert image to gray scale
gray = cv2.cvtColor(subtracted, cv2.COLOR_BGR2GRAY)

# location you recognize (using a mask)
mask = np.zeros(gray.shape[:2], dtype = "uint8")
(cX, cY) = (gray.shape[1] /2, gray.shape[0] / 2)
cv2.rectangle(mask, (cX - 160, cY - -120), (cX + 390 , cY - -205), 255, -1) # change this line for other image (second mask)
masked = cv2.bitwise_and(gray, gray, mask = mask)
cv2.imshow("Mask Applied to Image22", masked)
cv2.waitKey(0)

# blur the image, find edges, and then find contours along
# the edged regions
blurred = cv2.GaussianBlur(masked, (5, 5), 0)
edged = cv2.Canny(blurred, 30, 150)
(_, cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

# sort the contours by their x-axis position, ensuring
# that we read the numbers from left to right
cnts = sorted([(c, cv2.boundingRect(c)[0]) for c in cnts], key = lambda x: x[1])

for (c, _) in cnts[0:]:
        (x, y, w, h) = cv2.boundingRect(c)
        print(x,y,w,h)
        if w >= 7 and h >= 28 and w < 80:
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 1)
                cv2.imshow('image',image)
                cv2.waitKey(0)

exit()


contours = []
for (c, _) in cnts[0:]:
        (x, y, w, h) = cv2.boundingRect(c)
        if w >= 7 and h >= 28 and w < 80:
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 1)
                cv2.imshow('image',image)
                cv2.waitKey(0)
                contours.append((x,y,w,h))
                (prex, prey, prew, preh) = (x,y,w,h)
                break
#prec, _ = cnts[2]
#(prex, prey, prew, preh) = cv2.boundingRect(prec)
#cv2.rectangle(image, (prex, prey), (prex + prew, prey + preh), (0, 0, 255), 1)
#cv2.imshow('image',image)
#cv2.waitKey(0)
#contours.append((prex, prey, prew, preh))
for (c, _) in cnts[3:]:
#	# compute the bounding box for the rectangle
	(x, y, w, h) = cv2.boundingRect(c)
        if abs(x-prex)<10 and abs(y-prey)<10 and abs((x+w)-(prex+prew))<10 and abs((y+h)-(prey+preh))<10:
                continue
        elif x>=prex and y>=prey and (x+w)<=(prex+prew) and (y+h)<=(prey+preh):
                continue 
        else:
                #cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 1)
                #cv2.imshow('image',image)
                #cv2.waitKey(0)
                (prex, prey, prew, preh) = (x,y,w,h)
                contours.append((x,y,w,h))
print(len(contours))
# loop over the contours
for c in contours:
    # compute the bounding box for the rectangle
    (x, y, w, h) = c
    print(x,y,w,h)
        
        # if the width is at least 7 pixels and the height
        # is at least 20 pixels, the contour is likely a digit
    if w >= 7 and h >= 28:
        
            # crop the ROI and then threshold the grayscale
            # ROI to reveal the digit
            roi = gray[y:y + h, x:x + w]
            thresh = roi.copy()
            T = mahotas.thresholding.otsu(roi)
            thresh[thresh > T] = 255
            thresh = cv2.bitwise_not(thresh)
                
                # deskew the image center its extent
            thresh = dataset.deskew(thresh, 20)
            thresh = dataset.center_extent(thresh, (20, 20))  
            cv2.imshow("thresh", thresh)
            cv2.waitKey(0)
            #grayROI = msk.applymask(roi, RoadMSK)
            # extract features from the image and classify it
            hist = hog.describe(thresh)
            digit = model.predict([hist])[0]
            print "Creo que el numero es: %d" % (digit)
            
                # draw a rectangle around the digit, the show what the
                # digit was classified as
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 1)
            cv2.putText(image, str(digit), (x - 5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("image", image)
            #uncomment to audio
            #os.systme("say Match")
            #os.system ("say Coincide")
            # prees a key to continue
            cv2.waitKey(0)

#To use sounds 
#Audio (Just for OSX)
#os.system ("say Coincide")
#
#Audio (Other Systems)
#os.system("../sound/si.wav")


#other images mask
#f4.jpg
#cv2.rectangle(mask, (cX - 134, cY - 145), (cX + 150 , cY - 115), 255, -1) #first mask
#cv2.rectangle(mask, (cX - 30, cY - -55), (cX + 220 , cY - -100), 255, -1) #second mask


#f11.jpg
#cv2.rectangle(mask, (cX - 300, cY - 450), (cX + 202 , cY - 390), 255, -1) #first mask
#cv2.rectangle(mask, (cX - 235, cY - 30), (cX + 340 , cY - -30), 255, -1) #second mask

#f14.jpg
#cv2.rectangle(mask, (cX - 200, cY - 285), (cX + 170 , cY - 225), 255, -1) #first mask
#cv2.rectangle(mask, (cX - 00, cY - 20), (cX + 450 , cY - -20), 255, -1) #second mask

#v.png
#cv2.rectangle(mask, (cX - 135, cY - 90), (cX + 102 , cY - 30), 255, -1) #first mask
#cv2.rectangle(mask, (cX - -20, cY - -165), (cX + 220 , cY - -235), 255, -1) #second mask

