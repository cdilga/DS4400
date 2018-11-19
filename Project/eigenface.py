import numpy as np
import os
import cv2

#Note that this code has been written for this project:

from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import scipy
import scipy.stats
import matplotlib.pyplot as plt



image_x = image_y = 256

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


#train a model

#produce results

#train another model

#produce results

#compare all results

images = loadImages(r'C:/Users/cdilg/Documents/NEU/DS4400/Project/croppedyale/yaleB01')

mean_face = np.mean(images, axis = 0)

rpca = PCA(100, svd_solver='randomized')
rpca.fit(images)

pca_images = rpca.transform(images)

#pltImage(rpca.components_[0,:])

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
def faceSolver(face):
    pca_face = rpca.transform(face.reshape(1, -1) - mean_face)

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

    threshold = 41000

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

#Now train with all the images