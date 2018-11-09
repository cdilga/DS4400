import pandas as pd
import numpy as np
import sklearn as sk
import sklearn.ensemble
import sklearn.metrics
import sklearn.feature_selection
import scipy.stats
import math
import random
import sklearn.linear_model
import matplotlib.pyplot as plt



def metrics(cm):
    tn, fp, fn, tp = cm.ravel()
    if ((tp+fp) == 0):
        precision = 0
    else:
        precision = tp/(tp+fp)

    if ((tp+fn)==0):
        recall = 0
    else:
        recall = tp/(tp+fn)
    
    if (np.sum(cm.ravel()) == 0):
        accuracy = 0
        error = 1
    else:
        accuracy = (tp+tn)/(np.sum(cm.ravel()))
        error = 1-accuracy

    return (precision, recall, accuracy, error)


def dispMetrics(cm, note=''):
    '''Show accuracy, error, precision, and recall'''

    precision, recall, accuracy, error = metrics(cm)

    note = note + r'\\' if note != '' else ''
    print(r'{}Accuracy={:.4f} \ \ \epsilon={:.4f} \\ Precision={:.4f} \ \ Recall={:.4f}'.format(
        note, accuracy, error, precision, recall))


headers = []

with open('headers.txt') as f:
    for line in f:
        headers.append(line.split(' ')[0])

df = pd.read_csv('spambase.txt', header=None, names=headers, index_col=False)


class o():
    pass


def split(percentage, dataframe):
    train = o()
    test = o()
    train.data = df.sample(frac=percentage, random_state=200)
    test.data = df.drop(train.data.index)

    train.x = train.data.iloc[:, 0:57]
    train.y = train.data['spam']

    test.x = test.data.iloc[:, 0:57]
    test.y = test.data['spam']
    return (test, train)


test, train = split(.75, df)

class Node:
    def __init__(self, attribute, cutoff, classification, left=None, right=None):
        '''
        Attribute is the attribute or column of the dataframe we are using to split.
        Cutoff is the value at which to pick less than or equal to. Thus or binary values this should be 0.

        Nodes is a list of nodes, if empty then this node is a leaf?
        Classification - only used if this node is a leaf
        
        '''
        self.left = left
        self.right = right
        self.cutoff = cutoff
        self.classification = classification
        self.attribute = attribute


class ID3:

    def __init__(self, features=None):
        self._features = features

    def _entropy(self, p):
        e = 0
        try:
            e = -p*math.log2(p)-(1-p)*math.log2(1-p)
        except:
            pass
        return e

    def _getFeatures(self, x):
        #get self._features random indicies out of x, bounded by the len of the columns
        if self._features == None:
            n = len(x.columns)
        elif x.shape[1] < self._features:
            n = x.shape[1]
        else:
            n = self._features
        return x.sample(n, axis=1)

    def _condEntropy(self, _x, _y, attr):
        '''x and y should both be list like'''
        partitionx = []
        y = np.array(_y)
        try:
            length = _x.shape[1]

        except:
            length = _x.shape[0]
        #NOTE SHOULD return either 0 or 1
        x = np.array(_x)
        for i in range(0, length, math.ceil(length/4)):
            if x[i] < attr:
                partitionx.append((x[i], y[i]))
        count = 0
        if len(partitionx) == 0:
            return 0
        for x, y in partitionx:
            if y == 1:
                count += 1
        
        return self._entropy(count/len(partitionx))

    def _singleEntropy(self, y):
        '''Not conditional entropy with consistent interface'''
        return self._entropy(np.count_nonzero(np.array(y))/len(y))

    def _ig(self, x, y, value):
        return self._singleEntropy(y) - self._condEntropy(x, y, value)

    def _argmaxIG(self, x, y):
        '''Takes a continuous series like x, and will find both the value and value of the highest information gain'''
        maximum = [0, 0]

        for value in range(int(np.min(x.unique())), int(np.max(x.unique())), math.ceil(np.max(x.unique())/4) ):
            ig = self._ig(x, y, value)
            if ig > maximum[0]:
                maximum = [ig, value]
        return maximum

    def _id3(self, x, y):
        '''x is a dataframe, which we will reduce each time'''

        if np.all(y) == 1:
                #if all are positive, return a new leaf with class 1
            if 1 in y:
                return Node(x.columns[0], '', 1, None, None)
            else:
                #if all are negative, return a new leaf with class 0
                return Node(x.columns[0], '', 0, None, None)

        #if x has 0 columns, then return a new leaf with class average?
        if len(x.columns) == 1:
            return Node('<=', '', np.median(y), None, None)

        #pick the attribute that best classifies examples
        result = []

        #this might not allow different features to be selected in subsequent rounds...
        for column in self._getFeatures(x).columns:
            result.append(self._argmaxIG(x[column], y))

        #result = x.apply(lambda x: self._argmaxIG(x, y))
        result = pd.DataFrame(result)

        colindex = result[[0]].idxmax(axis=0)[0]
        val = result.iloc[colindex, 1]

        #we need something that returns only the rows that have x[attribute] less than split, and greater than split
        #get a left x, and a right x with left less than the corresponding value inside of result at the same row index
        newdf = x.merge(pd.DataFrame(
            y, columns=['response']), left_index=True, right_index=True)
        #check this is keeping the row information intact

        left = newdf.loc[newdf[x.columns[colindex]] < val].drop(
            columns=newdf.columns[colindex])
        right = newdf.loc[newdf[x.columns[colindex]] >=
                          val].drop(columns=newdf.columns[colindex])
        leftx = left.iloc[:, :-1]
        rightx = right.iloc[:, :-1]

        lefty = left.iloc[:, -1]
        righty = right.iloc[:, -1]

        #also get a left y and a right y with the same splits
        #pass these on recursively

        node = Node(x.columns[colindex], val,
                    None, self._id3(leftx, lefty), self._id3(rightx, righty))

        return node

    def fit(self, x, y):
        '''
        x - an numpy arraylike with 2 dimensions
        y - some numpy array with 1 dimensions
        '''
        self._defaultLabel = scipy.stats.mode(y)
        self._tree = self._id3(x, y)

    def _decide(self, tree, x):
        if tree.left == None and tree.right == None:
            return tree.classification
        else:
            if x.loc[x.first_valid_index(), tree.attribute] < tree.cutoff:
                return self._decide(tree.left, x)
            else:
                return self._decide(tree.right, x)
    def _debug(self, tree, padding):
        if tree.left == None and tree.right == None:
            return
        else:
            print(padding + str(tree.cutoff))
            self._debug(tree.left, padding + '    ')
            self._debug(tree.right, padding + '    ')

    def debug(self):
        return self._debug(self._tree, '')
    def predict(self, x):
        #This will go down the tree and make decisions at each node, and take the left and right option
        return self._decide(self._tree, x)


class RandomForestCustom:
    def __init__(self, trees, features=None):
        '''We have the number of trees, and the number of features'''
        self._ntrees = trees
        self._features = features
        self._trees = []

    def _bag(self, x, y):

        retx = x.copy()
        rety = y.copy()
        for i in range(x.shape[0]):
            retx.iloc[i,:] = x.iloc[random.randint(0, x.shape[0]-1), :]
            rety.iloc[i] = y.iloc[random.randint(0, len(y)-1)]
        return (retx, rety)

    def fit(self, x, y):
        #We need to add some bagging step for each of the trees in our forest
        #take a sample of x until it has the same number of observations as x, with replacement (and the same for y)

        for i in range(self._ntrees):
            self._trees.append(ID3(self._features))
            bagged_x, bagged_y = self._bag(x, y)
            self._trees[i].fit(bagged_x, bagged_y)

    def predict(self, x):
        r = []
        for row in x.iterrows():
            result = np.zeros_like(self._trees)
            for i in range(self._ntrees):
                result[i] = self._trees[i].predict(x)
            #easy way to get a class out of a vote...
            r.append(np.median(result))
        return r
    
        


t = ID3()
x = pd.DataFrame({'f1': [1, 0, 1, 1, 1], 'f2': [
                 0.1, 0.2, 0.2, 0.8, 1], 'f3': [0, 1, 2, 3, 4]})

y = pd.DataFrame({'response': [1, 1, 0, 0, 1]})
#test.fit(x, y)
t.fit(train.x, train.y)
t.debug()
#print(test.predict(pd.DataFrame({'f1': [1], 'f2': [0.1], 'f3': [0]})), pd.DataFrame({'response': [1]}))


myrfc = RandomForestCustom(10)
myrfc.fit(train.x, train.y)
vals = myrfc.predict(train.x)
cm = sk.metrics.confusion_matrix(test.y, np.array(vals))
dispMetrics(cm, 'Custom Random Forest Classifier')

'''

test.fit(x, y)
print(test.predict(pd.DataFrame({'f1': [1, 1], 'f2': [0.1, 0.1], 'f3': [4, 4]})))

testrfc = RandomForestCustom(10, 10)
testrfc.fit(x, y)
print(testrfc.predict(pd.DataFrame(
    {'f1': [1, 0], 'f2': [0.1, 0], 'f3': [4, 1]})))


m = len(train.x.columns)
for k in [m, int(m/2), int(math.sqrt(m))]:
    for model in [RandomForestCustom(10, m), sklearn.ensemble.RandomForestClassifier(10, m)]:
        model.fit(train.x, train.y)
        cm = sk.metrics.confusion_matrix(test.y, model.predict(test.x))
        dispMetrics(cm, '{} Classifier Test for m = {}'.format(model, m))
        cm = sk.metrics.confusion_matrix(train.y, model.predict(train.x))
        dispMetrics(cm, '{} Classifier Training m = {}'.format(model, m))
m = int(math.sqrt(m))

for l in [10, 50, 100]:
    for model in [RandomForestCustom(l, m), sklearn.ensemble.RandomForestClassifier(l, m)]:
        model.fit(train.x, train.y)
        cm = sk.metrics.confusion_matrix(test.y, model.predict(test.x))
        dispMetrics(cm, '{} Classifier Test, trees = {}'.format(model, l))
        cm = sk.metrics.confusion_matrix(train.y, model.predict(train.x))
        dispMetrics(cm, '{} Classifier Training, trees = {}'.format(model, l))
'''
