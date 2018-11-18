import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

data = [[1, 2], [2, 3], [3,5], [3,6], [6, 7], [5,7], [5,9], [9,11]]
gamma = np.array(data)

def disp(data):


# average data

'''
Where 
phi - average
gamma - individual images
v - the eigen vector matrix
w - the eigen values corresponding to the eigenvectors


'''


#print(w)
#print(v)

#We now have our values (w) and vectors (v) where w is the linear factor which is scaled by v

#pick a difference, called l
print('psi[0]')
print(psi[0])
print('v[1]')
print(v[1])
print(np.matmul(v[1], psi[0]))
print(np.matmul(v[0], psi[0]))


#now suppose we have some m best eigenvectors from v. we want to multiply these out by each of the examples in 
class CustomPCA():
    def __init__(self, m):
        self._m = m

    def disp(self, u):
        plt.scatter(data[:, 0], data[:, 1], label='data')

        plt.xlabel('x label')
        plt.ylabel('y label')

        plt.title("Simple Plot")

        plt.legend()

        plt.show()

    def train(self, gamma, verbose = 0):
        self.dims = gamma.shape[1]
        '''Gamma is a 2d array, of (m x n) with m training examples and n dimensions'''
        if verbose = 1:
            print('psi')
            print(psi.shape)
            print('v')
            print(v.shape)
            #quickly sort the eigenvalues and then reshuffle the eigenvectors accordingly
            print("argsorting:")
            print(w)
            print(np.argsort(-w))
            solver = PCA()
            result = solver.fit_transform(gamma)
            print(result)
            self.disp(u)

        phi = np.mean(gamma, axis=0)
        psi = gamma - phi
        w, v = np.linalg.eig(np.matmul(psi.T, psi))
        v = v[:,np.argsort(-w)][:,:self._m]

        u = np.matmul(psi, v)

        self.w = w
        self.v = v

        return u
    def transform(self, points):
        '''Takes some points and returns the relevant PCA components'''
        u = np.matmul(points, self.v)
    
    #those are converted
    



    #print('U matrix: \n {}'.format(U))

    #matrix_w = np.hstack((eig_pairs[0][1].reshape(3, 1), eig_pairs[1][1].reshape(3, 1)))
    #print('Matrix W:\n', matrix_w)
    # now project onto principal components


