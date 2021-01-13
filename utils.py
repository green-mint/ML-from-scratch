import numpy as np
class StandardScalar():
    def __init__(self,with_std=True,with_mean=True):
        self.with_std = with_std
        self.with_mean = with_mean #too lazy to do this rn
    
    def fit(self,X,y=None):
        try:
            isinstance(X,np.ndarray)
        except:
            print(f"X must be type 'numpy.ndarray' but is of type {type(X)}")
        
        self.mean_ = np.mean(X,axis=0)
        self.var_ = np.var(X,sxis=0)
        self.scale_ = np.sqrt(self.var_)

    def transform(self,X,y=None):
        try:
            isinstance(X,np.ndarray)
        except:
            print(f"X must be type 'numpy.ndarray' but is of type {type(X)}")
        return (X - self.mean_)/self.scale_
    
    def fit_transform(self,X,y=None):
        self.fit(X,y)
        return self.transform(X,y)