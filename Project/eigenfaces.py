import numpy as np
import matplotlib.pyplot as plt

class CustomPCA():
    def __init__(self, m):
        self.m = m

    def disp(self, u):
        plt.scatter(u[:, 0], u[:, 1], label='data')

        plt.xlabel('x label')
        plt.ylabel('y label')

        plt.title("Simple Plot")

        plt.legend()

        plt.show()

    def fit(self, gamma, verbose = 0):
        self.dims = gamma.shape[1]
        '''Gamma is a 2d array, of (m x n) with m training examples and n dimensions'''
        

        phi = np.mean(gamma, axis=0)
        self.phi = phi                      # keep the average image
        psi = gamma - phi
        w, v = np.linalg.eig(np.matmul(psi.T, psi))
        sortindex = np.argsort(-w)

        self.components = v[:, sortindex]
        v = self.components[:, :self.m]

        w = w[sortindex][:self.m]

        u = np.matmul(psi, v)
        #normalize u
        norm = np.linalg.norm(u, axis=0)
        u = u / norm
        
        self.w = w
        self.u = u
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
            
        
        return u
    def transform(self, image):
        '''Takes some points and returns the relevant PCA components'''
        pass
    def getComponents(self):
        return self.u

    def projectFace(self, image):
        #TODO add in some errors for mismatched dimensions
        normalised_image = image - self.phi
        weights = np.dot(self.u.T, normalised_image)
        newImage = np.dot(self.u[:, :self.m], weights[:, :self.m])
        return newImage
