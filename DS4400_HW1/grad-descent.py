import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
                   
# Load the file
housing = pd.read_csv("data/train.csv", 
                      usecols=lambda x: x in ["price",
                                              "bedrooms",
                                              "bathrooms",
                                              "sqft_living",
                                              "sqft_lot",
                                              "floors",
                                              "waterfront",
                                              "view",
                                              "condition",
                                              "grade",
                                              "sqft_above",
                                              "sqft_basement",
                                              "yr_built",
                                              "yr_renovated",
                                              "lat",
                                              "long",
                                              "sqft_living15",
                                              "sqft_lot15"])
housing_test = pd.read_csv("data/test.csv",
                           usecols=lambda x: x in ["price",
                                                   "bedrooms",
                                                   "bathrooms",
                                                   "sqft_living",
                                                   "sqft_lot",
                                                   "floors",
                                                   "waterfront",
                                                   "view",
                                                   "condition",
                                                   "grade",
                                                   "sqft_above",
                                                   "sqft_basement",
                                                   "yr_built",
                                                   "yr_renovated",
                                                   "lat",
                                                   "long",
                                                   "sqft_living15",
                                                   "sqft_lot15"])

def standardize(features):
    #iterate through the features
    features = np.array(features)
    
    #Nice function which copies the shape of features into a new array
    scaled = np.empty_like(features)
    for i in range(len(features.T)):
        mu = features[:,i].mean()
        sd = features[:,i].std()

        for j in range(len(features[:,i])):
            scaled[j,i] = (features[j,i] - mu)/sd
            
    return scaled

housing_training_features = housing.copy().drop("price", axis=1)
housing_training_labels = housing.copy().loc[:,"price"]

housing_testing_features = housing_test.copy().drop("price", axis=1)
housing_testing_labels = housing_test.copy().loc[:, "price"]

class Grad_lin:
    # has a coeficients property
    # has a fit method
    
    def __init__(self, threshold, alpha):
        self._rss = 0
        self.n = 0
        self._tss = 0
        self._alpha = alpha
        self.threshold = threshold
    
    def get_alpha(self):
        return self._alpha
    
    def derivative(self, X, y, j):
        # TODO replace with NumPy optimised iterator
        d = 0
        for i in range(len(X)):
            # Good chance this X is wrong here:
            # If there are errors on this line, make sure you're getting the ROW i
            d += (self._theta.dot(X[i]) - y[i])*X[i, j]
        ret = (d*2)/len(X)
        return ret

    def h(self, x, theta):
        x = np.array([x]).T
        return theta.T.dot(x)

    def cost(self, X, y, theta):
        return np.sum(np.power(X.dot(theta) - y, 2))/len(X)

    def fit(self, X, y): 
        # while not converged
        intercept = np.ones(len(X))  
        limit = 100
        #dt is the difference of thetas
        X = np.column_stack((intercept, X))
        dt = self.threshold + 1
        y = np.array([y]).T
        n = 0
        last_cost = 0
        self._theta = np.zeros([X.shape[1], 1])

        # This is really hacky
        self._old_theta = self._theta + 1
        self._cost = np.zeros((limit, 1))
        while ((np.linalg.norm(self._theta - self._old_theta) > self.threshold) and (n < limit)):
            
            # TODO use numpy optimised iterator
            #print(X.shape)
            #print(y.shape)
            #print(self._theta.shape)
            #(X.dot(self._theta.T) - y)
            #self._theta = self._theta - self._alpha*2/len(X) * np.sum(, axis=0)
            self._old_theta = self._theta.copy()
            for j in range(len(self._theta)):
                diff = 0
                for i in range(len(X)):
                    h = self.h(X[i], self._old_theta)
                    diff += (h - y[i])*X[i, j]
                self._theta[j] = self._old_theta[j]-self._alpha*diff/len(X)
            self._cost[n, 0] = self.cost(X, y, self._theta)
            last_cost = self._cost[n-1, 0]
            print(self._cost[n, 0], end = ",")
            n += 1
        return self
    
    def coef(self):
        return self._theta[1:]
    
    def intercept(self):
        return self._theta[0]
    
    def _calc_rss(self, X, y, y_m):
        #we have to strip off all of the leading 1's
        for i in range(self.n):
            self._rss += (y[i] - self.predict(X[i,1:]))**2
            self._tss += (y[i] - y_m)**2
            
    def mse(self):
        return (self._rss / self.n)
        
    def rss(self):
        return self._rss
    
    def r2(self):
        return 1-(self._rss/self._tss)
    
    def predict(self, x):
        """Takes a list of features for a single point, not including a leading 1"""
        return self._theta.T.dot(np.insert(np.array(x), 0, 1)) 

print("Running")  
gradmod = Grad_lin(0.01, 0.1)
gradmod.fit(standardize(housing_training_features), housing_training_labels)

model = gradmod.predict(np.array(housing_testing_features.loc[0,:]))
actual = housing_testing_labels[0]
print(gradmod.coef())
print(gradmod.intercept())
print(model, actual)
