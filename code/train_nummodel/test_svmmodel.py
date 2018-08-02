import os
import cv2
import cPickle
from imagesearch.hog import HOG
from imagesearch import dataset

model = open('mysvm.cpickle').read()
model = cPickle.loads(model)
hog = HOG(orientations = 18, pixelsPerCell = (10, 10), cellsPerBlock = (1,1))#, transform_sqrt = True, block_norm="L2")

test_dir = 'add/'
i = 0 
for image_dir in os.listdir(test_dir):
    if os.path.isdir(test_dir + image_dir):
        image_class = int(image_dir)
        for image in os.listdir(test_dir+image_dir):
            im = cv2.imread(test_dir+image_dir+'/'+image)
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            thresh = dataset.center_extent(gray, (20, 20))
            hist = hog.describe(thresh)
	    digit = model.predict([hist])[0]
            if image_class == digit:
                i += 1
            else:
                print(test_dir+image_dir+'/'+image, digit)
print('accurary: ', i)
