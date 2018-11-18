import numpy as np
import os
import cv2

#Note that this code has been written for this project:

from sklearn.metrics import confusion_matrix
from sklearn.decomposition import RandomizedPCA
from sklearn.svm import SVC


#Resize one of the images to 16x16 and then export it to an image file

#Open image
#First lets make a thing that does pca


def resizeImage(image):
    return cv2.resize(image, (32, 32), 0, 0, cv2.INTER_AREA)

def loadImages(root):
    images = []
    for path, subdirs, files in os.walk(root):
        for name in files:
            if name[-4:] == '.pgm':
                image = cv2.imread(os.path.join(path, name))
                image = resizeImage(image)
                images.append(image)
    return np.array([image[:, :, 0].flatten() for image in images])

def showImage(image):
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', 600, 600)
    cv2.imshow('image', cv2.convertScaleAbs(image.reshape(32, 32)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def actualImage(image):
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', 600, 600)
    cv2.imshow('image', cv2.convertScaleAbs(image))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#train a model

#produce results

#train another model

#produce results

#compare all results

images = loadImages(r'C:/Users/cdilg/Documents/NEU/DS4400/Project/croppedyale/yaleB01')

rpca = RandomizedPCA(10)
rpca.fit(images)


#I have no idea why the dimensionality here is wrong. 
# It's likely that some optimisation step has left this a different dimension to the ones we were expecting

#I currently have a lack of understanding of how the PCA.components_ actually works, and how these components would fit into 
# the paper which I have been reading, as I would have expected these components to be simply u, however the dimensionality is 
# wrong, and either there is an error in the implementation or u is not the 'components' (I suspect the implementation is wrong
# because the paper refers to u as the eigenfaces, which should be a face representation and therfore each face should have a 
# dimensionality of (1, 1850))

actualImage(cpca.getComponents().reshape((cpca.m, 32, 32))[0])

