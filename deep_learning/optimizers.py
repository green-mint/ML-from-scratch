import numpy as np


class Optimizer():
    def __init__(self,learning_rate=0.01):
        self.learning_rate = learning_rate

class SGD(Optimizer):
    def __init__(self,learning_rate):
        self.learning_rate = learning_rate
        self.name = 'SGD'

    def optimize(self,W,b,gradients):
        """
        Arguments:
        W -- weight matrix
        b -- bias vector
        gradients -- dict of gradients dW and db

        Return:
        W -- updated weight matrix
        b -- updated bias vector
        """
        dW = gradients['dW']
        db = gradients['db']
        W = W - self.learning_rate * dW
        b = b - self.learning_rate * db
        return W,b



class Momentum(Optimizer):
    """
    Parameters:
        beta -- Exponential decay hyperparameter for the first moment estimates
    """
    def __init__(self,beta=0.9):
        super().__init__()
        self.beta = beta
        self.name = 'momentum'
    
    def optimize(self,W,b,gradients,V,t):
        """updates the parameters using gradient descent momentum

        Arguments:
        W -- weight matrix
        b -- bias vector
        gradients -- dict of gradients dW and db

        Return:
        W -- updated weight matrix
        b -- updated bias vector
        """
        dW = gradients['dW']
        db = gradients['db']

        #Calculating moving average of the gradients
        V['VdW'] = self.beta*V['VdW'] + (1-self.beta)*dW
        V['Vdb'] = self.beta*V['Vdb'] + (1-self.beta)*db
        #applying bias correction
        VdW_corrected = V['VdW']/(1-self.beta**(t+1)) #the value of t start from 0 so t+1
        Vdb_corrected = V['Vdb']/(1-self.beta**(t+1))
        #updating the parameters
        W = W - self.learning_rate * VdW_corrected
        b = b - self.learning_rate * Vdb_corrected
        return W,b

class Adam(Optimizer):
    """
    Parameters:
        beta1 -- Exponential decay hyperparameter for the first moment estimates
        beta2 -- Exponential decay hyperparameter for the second moment estimates
    """
    def __init__(self,beta1=0.9,beta2=0.99):
        super().__init__()
        self.beta1 = beta1
        self.beta2 = beta2
        self.name = 'adam'

    def optimize(self,W,b,gradients,V,S,t):
        """updates the parameters using adam optimization
    
        Arguments:
        W -- weight matrix
        b -- bias vector
        gradients -- dict of gradients dW and db
        t -- the mini_batch index being used

        Return:
        W -- updated weight matrix
        b -- updated bias vector"""

        dW = gradients['dW']
        db = gradients['db']
    
        #Calculating moving average of the gradients
        V['VdW'] = self.beta1*V['VdW'] + (1-self.beta1)*dW
        V['Vdb'] = self.beta1*V['Vdb'] + (1-self.beta1)*db
        #applying bias correction
        VdW_corrected = V['VdW']/(1-self.beta1**(t+1))
        Vdb_corrected = V['Vdb']/(1-self.beta1**(t+1))

        #Calculating moving average of the square gradients
        S['SdW'] = self.beta2*S['SdW'] + (1-self.beta2)*(dW**2)
        S['Sdb'] = self.beta2*S['Sdb'] + (1-self.beta2)*(db**2)
        #applying bias correction
        SdW_corrected = S['SdW']/(1-self.beta2**(t+1))
        Sdb_corrected = S['Sdb']/(1-self.beta2**(t+1))

        #updating parameters
        W = W - self.learning_rate * VdW_corrected/np.sqrt(SdW_corrected + 1e-8)
        b = b - self.learning_rate * Vdb_corrected/np.sqrt(Sdb_corrected + 1e-8)
        return W,b