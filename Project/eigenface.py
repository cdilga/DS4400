#from skimage.transform import resize
import numpy as np
import os
#Resize one of the images to 16x16 and then export it to an image file

#Open image

import cv2
#First lets make a thing that does pca


def makeEigenFace(images):
    for face in images.iterrows():
        C = 
    #return eig


#train a model

#produce results

#train another model

#produce results

#compare all results

# A class that will be a image classifier.
#there needs to be a model
#there needs to be things that will preprocess everything
#something that does something with filepaths and images

def resizeImage(image):
    return cv2.resize(image, (16, 16), 0, 0, cv2.INTER_AREA)

def diffFaces(images):
    avg = np.mean(images, axis=0)
    normalised = images - avg
    return normalised
       

def loadImages(root):
    images = []
    for path, subdirs, files in os.walk(root):
        for name in files:
            if name[-4:] == '.pgm':
                image = cv2.imread(os.path.join(path, name))
                image = resizeImage(image)
                images.append(image)
    return np.array([image[:,:,0].flatten() for image in images])
#Read Image
#img = cv2.imread('croppedyale/yaleB01/yaleB01_P00A+000E+00.pgm')
#Remove the rgb data from the image
#img = img[:,:,0]
#Display Image

images = loadImages('croppedyale/yaleB01')
diffs = diffFaces(images)
cv2.imshow('image', makeEigenFace(diffs))
cv2.waitKey(0)
cv2.destroyAllWindows()

