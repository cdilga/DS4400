import pandas as pd
import numpy as np
import sklearn as sk
import sklearn.ensemble
import sklearn.metrics
import sklearn.feature_selection
import scipy.stats
headers = []
with open('headers.txt') as f:
    for line in f:
        headers.append(line.split(' ')[0])

df = pd.read_csv('spambase.txt', header=None, names=headers, index_col=False)


class o():
    pass


train = o()
test = o()

train.data = df.sample(frac=0.75, random_state=200)
test.data = df.drop(train.data.index)

train.x = train.data.iloc[:, 0:57]
train.y = train.data['spam']

test.x = test.data.iloc[:, 0:57]
test.y = test.data['spam']

rfc = sk.ensemble.RandomForestClassifier()

rfc.fit(train.x, train.y)

cm = sk.metrics.confusion_matrix(test.y, rfc.predict(test.x))

tn, fp, fn, tp = cm.ravel()

print(test.x)


class Node:
    def __init__(self, attribute, cutoff, classification, left = None, right = None):
        '''
        Attribute is the attribute or column of the dataframe we are using to split.
        Cutoff is the value at which to pick less than or equal to. Thus or binary values this should be 0.

        Nodes is a list of nodes, if empty then this node is a leaf?
        Classification - only used if this node is a leaf
        
        '''
        self.left = left
        self.right = right
        self.classification = classification
        self.attribute = attribute    

    def addNode(self, node):
        pass


class ID3:
    def __init__(self):
        pass

    def _id3(x, y):
        #if all are positive, return a new leaf with class 1
        #if all are negative, return a new leaf with class -1
        #if x has 0 columns, then return a new leaf with class average?

        #pick the attribute that best classifies examples
        #create node with that attribute.

        self._tree = Node()

    def fit(x, y):
        '''
        x - an numpy arraylike
        y - some numpy array with 2 dimensions
        '''
        self._defaultLabel = scipy.stats.mode(y)
        
        #list the attributes we have to pick from
        y.
        #attach an information gain to the attributes
        #pick the best one
        #create a node with best IG
        #repeat this process with less attributes available
        #

    def predict(x):
        self._tree
        #This will go down the tree and make decisions at each node, and take the left and right option


class RandomForestCustom:
    def __init__(self, trees, features):
        self.trees = trees
        self.features = features

#IG sklearn.feature_selection
