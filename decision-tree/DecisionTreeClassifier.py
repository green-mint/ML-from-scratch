from utils_classif import *

class DecisionTreeClassifier():
    
    def __init__(self,max_depth=None,min_samples=3,choose_k_lessthan_d=False):
        self.tree = None
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.column_headers = None
        self.choose_k_lessthan_d = choose_k_lessthan_d
    
    def train(self,X,y):
        '''The feature matrix(X) must be a data frame, so that the column
        headers may be extracted'''
        column_headers = X.columns
        feature_types = get_feature_types(X)
        self.counter = 0
        X = X.values
        y = y.values.reshape(y.size,-1)
        data = np.hstack((X,y)) #Converting the dataframes into an array
        
        self.tree = decision_tree_classification(data,self.min_samples,self.max_depth,column_headers,feature_types,self.counter,choose_k_lessthan_d=self.choose_k_lessthan_d)
        return self.tree
    
    def predict(self,X):
        '''The feature matrix(X) must be a data frame, so that the column
        headers may be extracted'''
        predictions = []
        for index in range(len(X)):
            example = X.iloc[index]
            prediction = single_predict(example,self.tree)
            predictions.append(prediction)
        predictions = np.array(predictions)
        return predictions

