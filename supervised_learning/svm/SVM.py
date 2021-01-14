import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers
from numpy import linalg

def linear_kernel(x1, x2):
    return np.dot(x1,x2)



def rbf_kernel(a, b, gamma): #Correct version to be implemented
    sqdist = np.sum(a**2,1).reshape(-1,1) + np.sum(b**2,1) - 2*np.dot(a, b.T)
    return np.exp(gamma * sqdist)

def poly_kernel(x1,x2,degree=3):
    return (1+np.dot(x1,x2))**degree

class SVM:
    def __init__(self,kernel=rbf_kernel,C=None,gamma=5.0,degree=3):
        self.kernel = kernel
        self.C = C
        if self.C is not None: self.C = float(C)
        self.gamma = gamma
        self.degree = degree
        self.sup_vecs = None #support vectors
        self.sup_mul = None #lagrange multipliers for the corresponding support vectors
        self.sup_labels = None #truth labels for the corresponding support vectors

    def train(self,X,y):
        self.n_samples = X.shape[0]
        # self.train_fm = X
        # self.labels = y

        #computing the kernel matrix(grams matrix)
        print(X.shape,y.shape)
        K = self.kernel(X,y.reshape(-1,1),self.gamma)

        #Converting into cvxopt format
        H = np.outer(y,y) * K
        P = cvxopt_matrix(H)
        q = cvxopt_matrix(-np.ones((self.n_samples, 1)))
        y = y.astype('float64') #y must be a array with float values for the next command
        A = cvxopt_matrix(y.T) 
        b = cvxopt_matrix(np.zeros(1))
        if self.C != None:
            G = cvxopt_matrix(np.vstack((-np.eye(self.n_samples),
                                        np.eye(self.n_samples))))
            h = cvxopt_matrix(np.vstack((np.zeros((self.n_samples,1)),
                                        np.ones((self.n_samples,1))*self.C)))
        else:
            G = cvxopt_matrix(-np.eye(self.n_samples))
            h = cvxopt_matrix(np.zeros(self.n_samples))
    
        #Setting solver parameters (change default to decrease tolerance) 
        cvxopt_solvers.options['show_progress'] = False
        cvxopt_solvers.options['abstol'] = 1e-10
        cvxopt_solvers.options['reltol'] = 1e-10
        cvxopt_solvers.options['feastol'] = 1e-10

        #Run solver
        sol = cvxopt_solvers.qp(P, q, G, h, A, b)
        alphas = np.array(sol['x'])

        #Selecting the set of indices S corresponding to non zero parameters
        S = [i for i in range(len(alphas)) if alphas[i] > 1e-4] #returns a boolean array with truth values representing support vectors
        #Computing b
        self.b = y[S[0]] - np.sum(y * alphas * K[S[0]].reshape(self.n_samples,1))

        self.sup_vecs = X[S]
        self.sup_mul = alphas[S]
        self.sup_labels = y[S]

    def predict(self,X):
        #Computing the kernel matrix
        K = np.zeros((X.shape[0],len(self.sup_vecs)))
        for i in range(len(X)):
            for j in range(len(self.sup_vecs)):
                if self.kernel == linear_kernel:
                    K[i,j] = self.kernel(x1=X[i],x2=self.sup_vecs[j])
                elif self.kernel == rbf_kernel:
                    K[i,j] = self.kernel(x1=X[i],x2=self.sup_vecs[j],sigma=self.sigma)
                elif self.kernel == poly_kernel:
                    K[i,j] = self.kernel(x1=X[i],x2=self.sup_vecs[j],degree=self.degree)    
        Z = self.sup_mul * self.sup_labels #element wise multiplication
        predictions = np.matmul(K,Z) + self.b
        return np.sign(predictions)


from matplotlib.colors import ListedColormap
def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    plt.figure(figsize=(9,6))
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    print('1')
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    print('2')
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    print('3')
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    print('4')
    pos = np.array([X[i] for i in range(X.shape[0]) if y[i] == 1])
    neg = np.array([X[i] for i in range(X.shape[0]) if y[i] == -1])
    plt.plot(pos[:,0],pos[:,1],'bo')
    plt.plot(neg[:,0],neg[:,1],'ro')
    print('5')
    # highlight test samples
    if test_idx:
        # plot all samples
        
        X_test, y_test = X[test_idx, :], y[test_idx]

        plt.scatter(X_test[:, 0],
                    X_test[:, 1],
                    c='',
                    alpha=1.0,
                    linewidths=1,
                    marker='o',
                    s=55, label='test set')

import pandas as pd 

datafile = './datasets/titanic.csv' 
df = pd.read_csv(datafile)

#Filling in missing age vales using the mean
df.Age = df.Age.fillna(value=np.mean(df.Age))

#converting the categorical values to numerical values
df = pd.get_dummies(df,columns=['Sex'],drop_first=True)

X = df.loc[:,['Age','Sex_male','Pclass','SibSp','Parch']].values
y = df.loc[:,'Survived'].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=10)

clf = SVM(kernel=rbf_kernel,gamma=0.1,C=1)
clf.train(X_train,y_train)
predictions = clf.predict(X_test,y_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,predictions)
print(cm)
