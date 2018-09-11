# import the necessary packages
from sklearn.svm import LinearSVC
from imagesearch.hog import HOG
import numpy as np
import cPickle
import mahotas
import cv2
import os


hog = HOG(orientations = 18, pixelsPerCell = (10, 10),
	cellsPerBlock = (1, 1))#, normalize = True)

def loadmnist(filedir):
    train_data = []
    train_labels = []
    for subdir in os.listdir(filedir):
        if os.path.isdir(filedir+subdir):
            for image in os.listdir(filedir+subdir):
                im = cv2.imread(filedir+subdir+'/'+image)
                gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                if not gray.shape == (80,60):
                    gray = cv2.resize(gray,(60,80),interpolation=cv2.INTER_CUBIC)
                T = mahotas.thresholding.otsu(gray)
                gray[gray > T] = 255
                gray[gray < T] = 0

                #image = dataset.deskew(image, 20)
                #image = dataset.center_extent(gray, (20, 20))
                hist = hog.describe(gray)

                train_data.append(hist)
                train_labels.append(int(subdir))
    return train_data, train_labels


train_data, train_labels = loadmnist('number/')

# train the model
model = LinearSVC(random_state = 42)
model.fit(train_data, train_labels)

# dump the model to file
f = open('models/num_char_new.cpickle', "w")
f.write(cPickle.dumps(model))
f.close()
