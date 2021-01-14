import numpy as np
import math

def initialize_minibatches(X,y,batch_size=64,seed=0):
    """ creates random mini batches (X_mini,y_mini )
    Arguments:
    X,y -- the feature matrix and the labels
    batch_size = number of examples in a sigle mini batch
    seed -- seed to prevent randomness
    """
    m = X.shape[1]
    np.random.seed(seed)
    mini_batches = []
    #applying the same shuffling techinques on X and y
    premutation = np.random.permutation(m)
    X_shuffled = X[:,premutation]
    y_shuffled = y[:,premutation]

    num_full_minibatches = math.floor(m/batch_size) #number of mini batches with size {batch_size}

    for t in range(num_full_minibatches):
        X_mini = X_shuffled[:,t*batch_size:(t+1)*batch_size]
        y_mini = y_shuffled[:,t*batch_size:(t+1)*batch_size]

        mini_batch = (X_mini,y_mini)
        mini_batches.append(mini_batch)
    
    if m % batch_size != 0: #the last minibatch gets the remaining elements(<batch_size)
        X_mini = X_shuffled[:,(t+1)*batch_size:] 
        y_mini = y_shuffled[:,(t+1)*batch_size:]

        mini_batch = (X_mini,y_mini)
        mini_batches.append(mini_batch)
    
    return mini_batches

def compute_cost(AL,y,layers,lambd,n_classes=2):
    m = y.shape[1]
    if n_classes == 2:
        cross_entropy = (np.sum(y*np.log(AL) + (1 - y)*np.log(1-AL)))/-m
    else:
        cross_entropy = np.sum(y * np.log(AL + 1e-8))/-m
    l2regularization = 0
    for layer in layers:
        l2regularization += np.sum(layer.W ** 2)
    return cross_entropy + (lambd/(2*m)) * l2regularization

def compute_dZL(AL,y):
    """Computes the dZ for the last layer(layer L)"""
    dZL = AL - y
    return dZL

def initialize_params_net(layers,X):
    prev_nerurons = X.shape[0]
    for layer in layers:
        layer.initialize_params_EWAs(prev_nerurons)
        prev_nerurons = layer.N

def propogate_forward_net(layers,X):
    A_prev = X
    for layer in layers:
        A = layer.propogate_forward(A_prev)
        A_prev = A
    AL = A_prev #AL here refers to the activations of the last layer, the output of the network
    return AL

def propogate_back_net(layers,y,AL,lambd,model):
    dA = compute_dZL(AL,y) #the first dA is actually dZL
    model.history['dZL'].append(dA)
    for layer in reversed(layers):
        dA_prev = layer.propogate_back(dA,lambd)
        dA = dA_prev

def update_parameters_net(layers,optimizer,t):
    for layer in layers:
        layer.update_params(optimizer,t)

def predict_probs_net(layers,X):
    probabilities = propogate_forward_net(layers,X)
    return probabilities

def predict_net(layers,X):
    predictions = predict_probs_net(layers,X)
    if predictions.shape[0] == 1:
        predictions[predictions <= 0.5] = 0
        predictions[predictions > 0.5] = 1
    else:
        predictions = np.where((predictions.max(axis=0,keepdims=True) == predictions),1,0)
    return predictions