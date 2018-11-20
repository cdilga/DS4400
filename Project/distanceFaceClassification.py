import numpy as np
import os
import cv2

#Note that this code has been written for this project:

from sklearn.metrics import confusion_matrix
import sklearn.cross_validation
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import scipy
import scipy.stats
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import sklearn.model_selection
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

import pickle

import eigenfaces

image_x = image_y = 32

verbose = 5
def resizeImage(image):
    return cv2.resize(image, (image_x, image_y), 0, 0, cv2.INTER_AREA)

def loadImages(root, extension = '.pgm'):
    images = []
    for path, subdirs, files in os.walk(root):
        for name in files:
            if name[-len(extension):] == extension:
                image = cv2.imread(os.path.join(path, name))
                image = resizeImage(image)
                images.append(image)
    return np.array([image[:, :, 0].flatten() for image in images])

def showImage(image):
    '''Note, has very weird scaling issues, often images end up black'''
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', 600, 600)
    cv2.imshow('image', image.reshape(image_x, image_y))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def pltImage(image):
    plt.imshow(np.resize(image, (image_x, image_y)), cmap=plt.cm.gray)
    plt.show()

#######################################################################
#
#   Evaluation of the distance metric based face classifier
#
#
#   This is a binary classifier and we get approx 80+% accuracy
#   Currently, the cutoff is manual, statistical methods forthcoming    
#                                                                       
#########################################################################



def evaluatedfc():
    dfc = eigenfaces.EuclideanFaceClassifier()


    print('Normal face images')
    faces = loadImages(
        r'C:/Users/cdilg/Documents/NEU/DS4400/Project/croppedyale/yaleB01')
    dfc.fit(faces)

    print('Non face images distance metric')
    not_faces = loadImages(
        r'C:/Users/cdilg/Documents/NEU/DS4400/Project/not-faces', '.jpg')[0:65, :]
    notFaceResults = dfc.predict(not_faces)

    print('Testing face images')
    unseen_images = loadImages(
        r'C:/Users/cdilg/Documents/NEU/DS4400/Project/croppedyale/yaleB02')
    unseenFaces = dfc.predict(unseen_images)

    fp = np.sum(notFaceResults)
    tn = not_faces.shape[0] - fp
    tp = np.sum(unseenFaces)
    fn = unseen_images.shape[0] - tp

    acc = (tp+tn)/(not_faces.shape[0] + unseen_images.shape[0])
    err = 1 - acc
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)

    if verbose > 0:
        print('Precision: {}'.format(precision))
        print('Recall: {}'.format(recall))
        print('Accuracy: {}'.format(acc))
        print('Error: {}'.format(err))

    return (precision, recall, acc, err)

#evaluatedfc()

'''
images = loadImages(r'C:/Users/cdilg/Documents/NEU/DS4400/Project/croppedyale/yaleB01')

mean_face = np.mean(images, axis = 0)

rpca = PCA(100, svd_solver='randomized')
rpca.fit(images - mean_face)

pca_images = rpca.transform(images - mean_face)

#pltImage(rpca.components_[0,:])
'''



'''
In the style of the original paper we will define a distance metric, 
below a cutoff value will classify a given source image as a face. 
This is in effect a binary classifier, and will tell us if what we're looking at is a face

In the style of the paper, we project the new image into the face space, and then compute euclidean distance


We use the initial set of 65 images, and then a mix of the second folder images and Caltech101 images of not faces

First we just get the distance metrics from the 'face' to it's face space. If it's a long distance, we haven't 
captured the variance well.

Similarly, we can use existing training examples in a similar way to a k-nn and classify. We'd need a k-nn modified
to allow a cutoff for an 'out of class' value, which we can implement.


'''
images = loadImages(r'C:/Users/cdilg/Documents/NEU/DS4400/Project/croppedyale/yaleB01')

rpca = PCA(100, svd_solver='randomized')
rpca.fit(images)

pca_images = rpca.transform(images)

def projectFaceView(face, pca):
    face = np.array(face)
        
    pca_face = pca.transform(face.reshape(1, -1))
    projected_face = np.matmul(pca_face, pca.components_)
    pltImage(projected_face)

def eigenFaceView(pca):
    eigFaces = pca.components_.reshape((len(pca.components_), image_x, image_y))
    pltImage(eigFaces[0])

def faceSolver(face):
    '''
    Warning - code is probably dead. 
    Do not use anything that normalises faces as pca is smart and already does this for us
    
    :O

    '''
    pca_face = rpca.transform(face.reshape(1, -1))

    #this step is rather important - we actually obtain the orig face back, but it's actually
    #just a linear combination of eigenvectors
    
    projected_face = np.matmul(pca_face, rpca.components_)

    #NOTE confusion around how to normalise new inputs, do we divide by the average of a class,
    # or all classes, or by the mean of the data itself?
    diff = face - projected_face

    return np.linalg.norm(diff)

def calculateDistance(imageset):

    #for training data
    distances = np.apply_along_axis(faceSolver, axis = 1, arr=imageset)
    #print(distances)
    print(scipy.stats.describe(distances))
    print(scipy.stats.tstd(distances))
    #for new images
    return distances

def toyDistanceClassifier():
    print('Normal face images')
    images = loadImages(
        r'C:/Users/cdilg/Documents/NEU/DS4400/Project/croppedyale/yaleB01')
    
    print('Testing images distance metric')
    calculateDistance(images)

    print('Non face images distance metric')
    not_faces = loadImages(
        r'C:/Users/cdilg/Documents/NEU/DS4400/Project/not-faces', '.jpg')[0:65,:]
    not_face_dist = calculateDistance(not_faces)

    print('Testing face images')
    unseen_images = loadImages(
        r'C:/Users/cdilg/Documents/NEU/DS4400/Project/croppedyale/yaleB02')
    test_face_dist = calculateDistance(unseen_images)
    #Define the distance cutoff to be a value, 34000 chosen after observing the accuracy

    threshold = 2700

    fp = np.sum(not_face_dist < threshold)
    tn = not_face_dist.size - fp
    tp = np.sum(test_face_dist < threshold)
    fn = test_face_dist.size - tp

    acc = (tp+tn)/(not_face_dist.size + test_face_dist.size)
    err = 1 - acc
    print('Precision: {}'.format(tp/(tp+fp)))
    print('Recall: {}'.format(tp/(tp+fn)))
    print('Accuracy: {}'.format(acc))
    print('Error: {}'.format(err))
toyDistanceClassifier()
