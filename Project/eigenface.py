import numpy as np
import os
import cv2

#Note that this code has been written for this project:
import eigenfaces


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

cpca = eigenfaces.CustomPCA(10)
cpca.fit(images)

#I have no idea why the dimensionality here is wrong. 
# It's likely that some optimisation step has left this a different dimension to the ones we were expecting

actualImage(cpca.getComponents().reshape((cpca.m, 32, 32))[0])

