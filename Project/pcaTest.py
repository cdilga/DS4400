import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

data = [[1, 2], [2, 3], [3,5], [3,6], [6, 7], [5,7], [5,9], [9,11]]
gamma = np.array(data)

'''
Where 
phi - average
gamma - individual images
v - the eigen vector matrix
w - the eigen values corresponding to the eigenvectors


'''

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

    def fit(self, gamma, verbose = 0):
        self.dims = gamma.shape[1]
        '''Gamma is a 2d array, of (m x n) with m training examples and n dimensions'''
        

        phi = np.mean(gamma, axis=0)
        psi = gamma - phi
        w, v = np.linalg.eig(np.matmul(psi.T, psi))
        v = v[:,np.argsort(-w)][:,:self._m]

        u = np.matmul(psi, v)

        self.w = w
        self.v = v

        if verbose == 1:
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
        
        return u
    def transform(self, points):
        '''Takes some points and returns the relevant PCA components'''
        #TODO add in some errors for mismatched dimensions
        u = np.matmul(points, self.v)
        return u

cpca = CustomPCA(2)
print(cpca.fit(gamma))
new_data = np.array([[1, 2, 3],[1,2, 3]])
print(cpca.transform(new_data))