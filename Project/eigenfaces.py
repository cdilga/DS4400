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

class EigenFacesLoader(object):
    def __init__(self, image_x, image_y, cache = ''):
        
        self._image_x = image_x
        self._image_y = image_y
        self._cache = cache

    def resizeImage(self, image):
        return cv2.resize(image, (image_x, image_y), 0, 0, cv2.INTER_AREA)


    def loadSomeImages(root, extension='.pgm'):
        images = []
        for path, subdirs, files in os.walk(root):
            for name in files:
                if name[-len(extension):] == extension:
                    image = cv2.imread(os.path.join(path, name))
                    image = self.resizeImage(image)
                    images.append(image)
        return np.array([image[:, :, 0].flatten() for image in images])

    def getEigenFaces(self):
        return self._pca.components_.reshape((len(pca.components_), image_x, image_y))

    def loadImages(self):
        '''Note, specifically for loading in images in the dir structure of the yale faces dataset'''
        root = self._cache
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
                    images = pd.DataFrame(loadSomeImages(root + subdir))
                    # Add a label column, with the name of the subdir
                    labels.extend([j]*len(images))
                    # Append this to a master DataFrame
                    imageset = imageset.append(images)

                labels = np.array(labels)
                break
            pickle.dump((imageset, labels), open('im.cache', 'wb'))
        self._images = imageset
        self._labels = labels
        return (imageset, labels)

class EigenFaceClassifier(object):
    def __init__(self, n_eigenfaces = 100, model = KNeighborsClassifier(5), size = 32):
        self._n_eigenfaces = n_eigenfaces
        self._model = model
        self._image_x = self._image_y = size
        

    def fit(self, x, y):
        '''
        x is a list of images
        y is labels or names
        
        Images should already be scaled appropriately, for doing this please use EigenFacesLoader
        which also has good caching
        '''
        train_x = x
        train_y = y
        self._rpca = PCA(self._n_eigenfaces, svd_solver='randomized')
        self._rpca.fit(train_x)
        pca_train_x = self._rpca.transform(train_x)

        self._model.fit(pca_train_x, train_y)
       

    def predict(self, x):
        '''Where x is a list of images'''
        return self._model.predict(self._rpca.transform(x))

    def pltImage(self, image):
        plt.imshow(np.resize(image, (self._image_x, self._image_y)), cmap=plt.cm.gray)
        plt.show()
        
    def showFace(self, n = 0):
        '''Default to show the principal axis of maximal variance'''
        eigFaces = self._rpca.components_.reshape(
            (len(self._rpca.components_), image_x, image_y))
        self._pltImage(eigFaces[n])

class EuclideanFaceClassifier(object):
    '''Will tell us if the thing is either a face or not, binary'''
    def __init__(self, n_eigenfaces = 100, cutoff = 41000, ):
        '''
        Distance metric chosen by hand to initialise to this value
        
        TODO: Actually choose distance metric using something like LDA
        to pick a cutoff point
        '''
        self._cutoff = cutoff
        self._n_eigenfaces = n_eigenfaces

    def faceSolver(self, face):
        '''
        Warning - code is probably dead. 
        Do not use anything that normalises faces as pca is smart and already does this for us
        
        :O

        '''
        pca_face = self._rpca.transform(face.reshape(1, -1))

        #this step is rather important - we actually obtain the orig face back, but it's actually
        #just a linear combination of eigenvectors

        projected_face = np.matmul(pca_face, self._rpca.components_)

        #NOTE confusion around how to normalise new inputs, do we divide by the average of a class,
        # or all classes, or by the mean of the data itself?
        diff = face - projected_face

        return np.linalg.norm(diff)

    def calculateDistance(self, imageset):
        #for training data
        distances = np.apply_along_axis(self.faceSolver, axis=1, arr=imageset)
        #print(distances)
        print(scipy.stats.describe(distances)) 
        print(scipy.stats.tstd(distances)) 
        #for new images
        return distances


    def fit(self, x, y = None, verbose = 2):
        self._rpca = PCA(self._n_eigenfaces, svd_solver='randomized')
        self._rpca.fit(x)

        print('Testing images distance metric')
        self._train_distances = self.calculateDistance(x)
    
    def predict(self, x):
        return self.calculateDistance(x) < self._cutoff

