import pandas as pd
import numpy as np
import sklearn as sk
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from IPython.display import display, Math
import math
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

#Read in data using the pandasreader
#Use pandas dataframe manipulation to sample 75% training and 25% testing

headers = []
with open('headers.txt') as f:
    for line in f:
        headers.append(line.split(' ')[0])
    
df = pd.read_csv('spambase.txt', header=None, names = headers, index_col = False)

train = df.sample(frac=0.75, random_state=200)
test = df.drop(train.index)

train.x = train.iloc[:, 0:57]
train.y = train['spam']

test.x = test.iloc[:, 0:57]
test.y = test['spam']


class Euclidean():
    '''This object will be injected into the classifier in order to calculate distances'''
    def __init__(self):
        pass
    
    def calc(self, x1, x2):
        '''xn can be numpy arrays of numerical values which we will compare the distance of'''
        #basically we square root the sum the difference between them to the power of 2
        #Get the differences
        
        x1 = np.array(x1)
        x2 = np.array(x2)
        
        d = x1 - x2
        
        #square all the elements - elementwise with **
        d = d**2

        return np.sum(d)
        
    
class Knn:
    def __init__(self, k, distance = Euclidean()):
        self.k = int(k)
        self.distmetric = distance
        
    def fit(self, x, y):
        '''As it's a knn, this doesn't do anything but store these values'''
        self.x = np.array(x)
        self.y = np.array(y)
        
    def predict_proba(self, x):
        #sort the points into distance order
        points = np.array(x)
        distances = np.zeros_like(points)
        
        probs = np.ones(len(points))

        for i in range(len(points)):
            #here we need to generate a classification
            #we want some datatype which has a probability at index i
            dist = np.empty((len(self.x), 2))
            for j in range(len(self.x)):
                #here we want some datatype which is a list of all the distances 
                dist[j,0] = self.distmetric.calc(points[i], self.x[j])
                
                #and keep the class in here too
                dist[j,1] = self.y[j]
            indicies = np.argsort(dist, axis=0)
            
            #get k of these indicies out for their classes, and multiply by 1/k
            
            #here we have the distances of all of the points to all of the training data
            #get the top k
            #weight each of these with 1/k weighting
            temp = self.y[indicies[0:self.k,0]]
            
            probs[i] = np.sum(temp * 1/self.k)
            #use this weight to determine probability of class 1 (we will assume binary classification here)
        #return probabilities that it's in class 1 in a 1d matrix
        return probs
        
    def predict(self, x, threshold = 0.5):
        #use predictproba to do all the real work and set a threshold arbitrarily
        probs = self.predict_proba(x)
        predictions = np.ones(len(x))
        try:
            for i in range(len(x)):
                predictions[i] = int(probs[i] > threshold)
        except:
            print(probs)
            print(i)
            print(probs[i])
        return predictions


class EfficientKnn:
    def __init__(self, k):
        self.k = int(k)

    def fit(self, x, y):
        '''As it's a knn, this doesn't do anything but store these values'''
        self.x = np.array(x)
        self.y = np.array(y)

    def predict_proba(self, x):
        #sort the points into distance order
        points = np.array(x)
        probs = np.ones(len(points))

        dist = np.empty((len(points), len(self.x)))
        for i in range(len(points)):
            dist[i,:] = np.sum((points - self.x[i,:])**2, axis = 0)

        indicies = np.argsort(dist, axis=0)
        temp = self.y[indicies[0:self.k, 0]]
        probs = np.sum(temp * 1/self.k, axis = 0)
        return probs

    def predict(self, x, threshold=0.5):
        #use predictproba to do all the real work and set a threshold arbitrarily
        probs = self.predict_proba(x)
        predictions = np.ones(len(x))
        for i in range(len(x)):
            predictions[i] = int(probs[i] > threshold)
        return predictions


models = []

customknn = Knn(3)
eff = EfficientKnn(3)
libknn = KNeighborsClassifier(3)

x = np.array([[1, 2, 3,], [1, 2, 4], [4, 4, 4]])
y = np.array([1, 1, 0])
for model in [customknn, libknn, eff]:
    model.fit(x, y)
    print(model, model.predict(np.array([[1, 2, 3], [1, 3, 3]])))


import time
start = time.time()

libknn.fit(train.x, train.y)
libknn.predict(test.x)
print("Library: ", time.time() - start)
start = time.time()

customknn.fit(train.x, train.y)
customknn.predict(test.x)
print("Custom: ", time.time() - start)


'''
for i, m in enumerate([Knn, KNeighborsClassifier]):
    kvals = range(3, 5)
    kacc = []
    kerr = []
    for k in kvals:
        model = m(k)
        model.fit(train.x, train.y)

        #Predict and record accuracy metrics
        predicted = model.predict(test.x)
        actual = test.y
        cm = confusion_matrix(predicted, actual)
        tn, fp, fn, tp = cm.ravel()
        kacc.append((tp+tn)/np.sum(cm.ravel()))
        kerr.append(1-(tp+tn)/np.sum(cm.ravel()))
    models.append([kacc, kerr])

print(models)
'''
