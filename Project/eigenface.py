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
#########################################################################
#
#   Real Eigenfaces classifier, using some models
#   
#   Uses all the images in the set for training, however the original
#   paper found that the best results were not always had with all 
#   images in the training set
#
#########################################################################


def prae(confusionMat):
    tn, fp, fn, tp = confusionMat.ravel()
    acc = (tp+tn)/(not_face_dist.size + test_face_dist.size)
    err = 1 - acc
    if verbose > 0:
        print('Precision: {}'.format(tp/(tp+fp)))
        print('Recall: {}'.format(tp/(tp+fn)))
        print('Accuracy: {}'.format(acc))
        print('Error: {}'.format(err))
    return (tp/(tp+fp), tp/(tp+fn), acc, err)

def loadAllImages(root):
    '''Note, specifically for loading in images in the dir structure of the yale faces dataset'''
    try:
        if(verbose > 1):
            print('Loading images')
        imageset, labels = pickle.load(open('im.cache', 'rb'))

    except FileNotFoundError:
        if (verbose > 1):
            print('Falling back to loading manually')
        imageset = pd.DataFrame()
        labels = []
        for path, subdirs, files in os.walk(root):
            for j, subdir in enumerate(subdirs):
                # Create a dataframe with the images in it
                images = pd.DataFrame(loadImages(root + subdir))
                # Add a label column, with the name of the subdir
                labels.extend([j]*len(images))
                # Append this to a master DataFrame
                imageset = imageset.append(images)

            labels = np.array(labels)
            break
        pickle.dump((imageset, labels), open('im.cache', 'wb'))
    return (imageset, labels)

#TODO replace this with the class for loading in the images
images, labels = loadAllImages(r'C:/Users/cdilg/Documents/NEU/DS4400/Project/croppedyale/')


splitter = sklearn.model_selection.StratifiedKFold(3)

results = []
i = 0
for train, test in splitter.split(images, labels):
    i += 1

    print('Training iteration {}'.format(i))
    train_x = images.iloc[train,:]
    train_y = labels[train]

    knn = KNeighborsClassifier(5)
    lda = LinearDiscriminantAnalysis(solver='svd')
    svc = SVC()
    #eigenFaceView(self._rpca)
    #projectFaceView(train_x.iloc[50,:], self._rpca)

    efc = eigenfaces.EigenFaceClassifier(model = lda)
    efc.fit(train_x, train_y)

    test_x = images.iloc[test,:]
    
    test_y = labels[test]
    predicted_y = efc.predict(test_x)

    metrics = sklearn.metrics.classification_report(test_y, predicted_y)


    results.append(metrics)
    print(metrics)

#pd.DataFrame(results).to_csv(open('svc_prae.csv', 'w'))



