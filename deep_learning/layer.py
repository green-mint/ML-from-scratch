import numpy as np
import math
from layer_utils import *

class Dense():
    """
    A layer object that reperesents one layer of neurons with its independent parameters

    Parameters:
    neurons -- number of neuron in the layer
    activation -- activation function used to calculate the layer output, relu or sigmoid
    keep_prob -- fraction of neurons to keep alive during every iteration

    Attributes:
    W -- The weight matrix for the layer
    b -- The bias vector for the layer
    """

    def __init__(self, neurons=2, activation='relu',keep_prob=1):
        self.N = neurons
        self.activation = activation        
        self.keep_prob = keep_prob
        self.W = None
        self.b = None
        self.cache = None
        self.gradients = None
        self.V = {} #dict for moving average of the gradients
        self.S = {} #dict for moving average of the square gradients

    def initialize_params_EWAs(self,prev_neurons):
        """ 
        Initializes the parametres W and b for the layer and the Exponetially Weighted Averages i.e.
        VdW,Vdb,SdW,Sdb

        W.shape == (# of neurons in current layer, # of neurons(features) in the previous layer)
        b.shape == (# of neurons in current layer,1)

        VdW.shape == SdW.shape == W.shape
        Vdb.shape == Sdb.shape == b.shape

        Arguments:
        prev_neurons -- number of neurons in the previous layer
        """
        #intializing the parameters of the layer
        self.W = np.random.randn(self.N,prev_neurons) * (np.sqrt(2/prev_neurons))
        self.b = np.zeros((self.N,1))

        #initalizing the exponentially weighted averages of gradients
        self.V['VdW'] = np.zeros(self.W.shape)
        self.V['Vdb'] = np.zeros(self.b.shape)

        #initalizing the exponentially weighted averages of square gradients
        self.S['SdW'] = np.zeros(self.W.shape)
        self.S['Sdb'] = np.zeros(self.b.shape)

    def propogate_forward(self,A_prev):

        """
        Propogates forward through the layer, computes the activation for the current layer and stores cache
        
        Arguments:
        A_prev -- Activations for the previous layer
        
        Returns:
        A -- activations for the current layer
        """
        A, self.cache = propogate_forward_layer(A_prev,self.W,self.b,self.activation,self.keep_prob)
                           # self.cache is a list i.e. [A_prev,Z,D]

        return A

    def propogate_back(self,dA,lambd):

        """
        Propogates back through the layer, computes the gradient of activations of previous layer;
        using the gradient of activations for the current layer(obtained by back prop in the l+1th;
        layer) and the cache of the curren layer

        Arguments:
        dA -- Gradient of activations for the current layer

        Returns:
        dA_prev -- gradient of activations for the previous layer 
        """
        
        dA_prev,self.gradients = propogate_back_layer(dA,self.W,self.cache,self.activation,lambd,self.keep_prob)
                # self.gradients is a dict of gradients i.e. {'dw','db'}
        return dA_prev
    

    def update_params(self,optimizer,t):
        """ Updates the parametres of the layer according to the optimizer provided
        
        Arguments:
        optimizer -- optimizer object used for optimization i.e. adam,sgd,momentum
        t -- the mini_batch index being used
        """
        
        if optimizer.name == 'SGD':
            self.W,self.b = optimizer.optimize(self.W,self.b,self.gradients)
        elif optimizer.name == 'momentum':
            self.W,self.b = optimizer.optimize(self.W,self.b,self.gradients,self.V,t)
        elif optimizer.name == 'adam':
            self.W,self.b = optimizer.optimize(self.W,self.b,self.gradients,self.V,self.S,t)

class Dropout():
    def __init__(self,keep_prob=1):
        self.keep_prob = keep_prob
        pass
