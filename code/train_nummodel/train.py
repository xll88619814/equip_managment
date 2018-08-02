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

# python train.py --dataset data/digits.csv --model models/svm.cpickle

# import the necessary packages
from sklearn.svm import LinearSVC
from pyimagesearch.hog import HOG
from pyimagesearch import dataset
import numpy as np
import argparse
import cPickle
import mahotas
import cv2
import os

def readnewdigits(imagepath):
    digits = []
    target = []
    for i in range(0,10):
        for image in os.listdir(imagepath+str(i)+'/'):
            im = cv2.imread(imagepath+str(i)+'/'+image)
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            T = mahotas.thresholding.otsu(gray)
            gray[gray > T] = 255
            gray[gray < T] = 0
            digits.append(gray)
            target.append(i)
    return digits, target 


def loadmnist(filedir):
    train_data = []
    train_labels = []
    for subdir in os.listdir(filedir):
        if os.path.isdir(filedir+subdir):
            for image in os.listdir(filedir+subdir):
                im = cv2.imread(filedir+subdir+'/'+image)
                gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                T = mahotas.thresholding.otsu(gray)
                gray[gray > T] = 255
                gray[gray < T] = 0
                train_data.append(gray)
                train_labels.append(int(subdir))
    return train_data, train_labels


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
#ap.add_argument("-d", "--dataset", required = True,
#	help = "path to the dataset file")
ap.add_argument("-m", "--model", required = True,
	help = "path to where the model will be stored")
args = vars(ap.parse_args())

# load the dataset and initialize the data matrix
#(digits, target) = dataset.load_digits(args["dataset"])
train_data, train_labels = loadmnist('/home/xll/trash/dataset/train_mnist/mnist_images/')
data = []
#print(digits.shape, type(target))
# initialize the HOG descriptor
hog = HOG(orientations = 18, pixelsPerCell = (10, 10),
	cellsPerBlock = (1, 1))#, normalize = True)

# loop over the images
for image in train_data:
    # deskew the image, center it
    image = dataset.deskew(image, 20)
    image = dataset.center_extent(image, (20, 20))
    # describe the image and update the data matrix
    hist = hog.describe(image)
    data.append(hist)

newdata, target = readnewdigits('add_images/')  
for i, image in  enumerate(newdata):
    image = dataset.deskew(image, 20)
    image = dataset.center_extent(image, (20, 20))
    hist = hog.describe(image)
    data.append(hist)
    train_labels.append(target[i])

# train the model
model = LinearSVC(random_state = 42)
model.fit(data, train_labels)

# dump the model to file
f = open(args["model"], "w")
f.write(cPickle.dumps(model))
f.close()
