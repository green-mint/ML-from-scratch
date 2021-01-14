import numpy as np
import random

class K_means:

    ''' rand_inits: # of times to run the algorithm with random initialization
        of the centroids in order to avoid local optima, default=1'''

    def __init__(self,clusters=3,max_iter=300,rand_inits=1):
        self.k = clusters
        self.centroids = None
        self.max_iter = 300
        self.rand_inits = 1
    
    def train(self,X):
        m,n = X.shape
        for z in range(self.rand_inits):

            #Intializes the centroids randomly
            self.centroids = []
            rand_indexes = random.sample(range(len(X)),k)
            for i in rand_indexes:
                self.centroids.append(X[i])
            
            #Optimization
            for b in range(self.max_iter):
                # Cluster assignment step: Assign each data point to the
                # closest centroid. idx[i] corresponds to cˆ(i), the index
                # of the centroid assigned to example i
                idx = np.zeros((m,1)) 
                for i in range(m):
                    dp = X[i]
                    distances = []
                    for j in range(len(self.centroids)):
                        distances.append(np.linalg.norm(X[i]-self.centroids[j]))
                    idx[i] = distances.index(min(distances))

                # Move centroid step: Compute means based on centroid
                # assignments
                new_centeroids = []
                for i in range(k):
                    a = np.array([X[j] for j in range(m) if idx[j] == i])
                    new_centeroids.append(np.sum(a,axis=0)/len(a))
                self.centroids = new_centeroids
    def predict(self, X):
        predictions = []
        for i in range(X.shape[0]):
            distances = []
            for j in range(len(self.centroids)):
                distances.append(np.linalg.norm(X[i]-self.centroids[j]))
                prediction = distances.index(min(distances))
            predictions.append([prediction])

