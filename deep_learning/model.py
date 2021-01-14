from model_utils import *
from layer import *
from optimizers import *

class MLP():
    def __init__(self,lambd=0,n_classes=2,optimizer='adam',seed=0):
        """
        A multilayed perceptron model

        Arguments:
        learning_rate -- step size
        epochs -- total number of iterations through the whole training set
        lambd -- l2 regularization parameter, directly propotional to regularization
        batch_size -- size of one training mini batch
                if batch_size == 1, corresponds to stochastic gradient descent
                if batch_size == m, corresponds to batch gradient descent
                else, corresponds to mini batch gradient descent
        optimizer -- the algorithm used for optimization. i.e. adam, gradient descent, momentum
        beta1 -- Exponential decay hyperparameter for the first moment estimates
        beta2 -- Exponential decay hyperparameter for the second moment estimates
        seed -- seed for randomness
        """
        if type(optimizer) == str:
            if optimizer == 'adam':
                self.optimizer = Adam()
            elif optimizer == 'momentum':
                self.optimizer = Momentum()
            elif optimizer == 'SGD':
                self.optimizer = SGD()
        else:
            self.optimizer = optimizer

        self.lambd = lambd    
        self.layers = []
        self.seed = seed
        self.n_classes = n_classes
        self.history = {'dZL':[]}
    
    def add_layer(self,layer_object):
        """
        Arguments:
        kwargs -- Keyword parameters for Layer() object
            neurons -- Number of neurons in the layer
            activation -- Activation function used for the layer [relu,sigmoid]
            keep_prob -- fraction of neurons to keep alive during every iteration
        """
        self.layers.append(layer_object)

    def fit(self,X,y,batch_size=32,epochs=100):
        m = X.shape[1]

        initialize_params_net(self.layers,X)
        mini_batches = initialize_minibatches(X,y,batch_size=batch_size,seed=self.seed)

        for epoch in range(epochs): 
            cost_batch = 0

            for t in range(len(mini_batches)):
                X_mini = mini_batches[t][0]
                y_mini = mini_batches[t][1]
                AL = propogate_forward_net(self.layers,X_mini)
                cost_batch += compute_cost(AL,y_mini,self.layers,self.lambd,self.n_classes)
                propogate_back_net(self.layers,y_mini,AL,self.lambd,self)
                update_parameters_net(self.layers,self.optimizer,t)
            cost_epoch = cost_batch/m

            print(f'Cost: {cost_epoch}, Accuracy: {self.accuracy(X,y)} ')
    
    def predict(self,X):
        return predict_net(self.layers,X)
    
    def accuracy(self,X,y):
        predictions = self.predict(X)
        if self.n_classes == 2:
            return np.mean(predictions == y)
        else:
            return ((predictions == y).sum(axis=0) == self.n_classes).sum()/X.shape[1]