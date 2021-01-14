import scipy as s
import matplotlib.pyplot as plt
import numpy as np 


train_X, train_Y, test_X, test_Y = np.loadtxt('train_X'),np.loadtxt('train_Y'),np.loadtxt('test_X'),np.loadtxt('test_Y')

train_Y = train_Y.reshape(1,-1)
test_Y = test_Y.reshape(1,-1)



from .. import model
from layer import Dense

mod = model.MLP(optimizer="adam")
mod.add_layer(Dense(neurons=20))
mod.add_layer(Dense(neurons=7))
mod.add_layer(Dense(neurons=5))
mod.add_layer(Dense(neurons=1,activation='sigmoid'))


costs = mod.fit(train_X, train_Y)

print(mod.accuracy(train_X,train_Y))
print(mod.accuracy(test_X,test_Y))

