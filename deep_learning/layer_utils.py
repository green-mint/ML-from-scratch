import numpy as np

def relu(Z):
    return np.maximum(0,Z)
    
def sigmoid(Z):
    return 1/(1+np.exp(-Z))

def softmax(Z):
    t = np.exp(Z-np.max(Z))
    return t/np.sum(t,axis=0)

def relu_der(Z): #derivative of relu
    dZ = Z 
    dZ[Z <= 0] = 0
    dZ[Z > 0] = 1
    return dZ

def sigmoid_der(Z): #derivative of sigmoid
    s = sigmoid(Z)
    dZ = s * (1-s)
    return dZ

def compute_dropout(A,keep_prob): 
    """computes the probability vector where eah entry corresponds to a neuron being shut down
    in the layer"""
    D = np.random.rand(A.shape[0],1)
    D = (D < keep_prob).astype(int)
    return D

def propogate_forward_layer(A_prev,W,b,activation,keep_prob):
    """
    Propogates forward through the layer, computes the activation for the current layer and stores cache
    
    Arguments:
    A_prev -- Activations for the previous layer
    W -- weight matrix of the current layer
    b -- bias vector for the current layer
    activation -- stored as a string, relu or sigmoid
    
    Returns:
    A -- activations for the current layer 
    cache -- a list [A_prev,Z,D]
    """
    Z = (W @ A_prev) + b
    if activation == 'relu':
        A = relu(Z)
    elif activation == 'sigmoid':
        A = sigmoid(Z)
    elif activation == 'softmax':
        A = softmax(Z)
    #shutting down (1-keep_prob) fraction of neurons, the D vector used in forward prop and back prop
    #is same so same neurons are shut in forward and backward prop of a single iteration
    D = compute_dropout(A,keep_prob)
    A = A * D
    A /= keep_prob

    cache = [A_prev,Z,D]
    return A,cache


def propogate_back_layer(dA,W,cache,activation,lambd,keep_prob):
    """
    Propogates back through the layer, computes the gradient of activations of previous layer;
    using the gradient of activations for the current layer and the cache
    
    Arguments:
    dA -- Gradient of activations for the current layer
    W -- weight matrix of the current layer
    activation -- stored as a string, relu or sigmoid
    
    Returns:
    dA_prev -- gradient of activations for the previous layer 
    gradients -- a dict of gradients i.e. {'dw','db'}
    """
    A_prev,Z,D = cache
    m = A_prev.shape[1] #number of examples, to be used for normalization

    #shutting down (1-keep_prob) fraction of neurons, the D vector used in forward prop and back prop
    #is same so same neurons are shut in forward and backward prop of a single iteration    
    dA = dA * D
    dA /= keep_prob
    
    if activation == 'relu':
        dZ = dA * relu_der(Z)
    elif activation == 'sigmoid':
        dZ = dA
    elif activation == 'softmax':
        dZ = dA
    dA_prev = W.T @ dZ
    dW = (dZ @ A_prev.T)/m + lambd * W/m
    db = (np.sum(dZ,axis=1,keepdims=True))/m

    gradients = {
        'dW':dW,
        'db':db
    }

    return dA_prev,gradients