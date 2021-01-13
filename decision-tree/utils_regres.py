import numpy as np
import pandas as pd

def create_leaf(data):
    '''Returns the most frequent label occuring in the
        dependent variable data thus creating a leaf node'''
    y = data[:,-1]
    label = np.mean(y)
    return label

def get_feature_types(df):
    n_unique_value_threshold = 5
    feature_types = []
    for index in range(df.shape[1]):
        datatype = str(df.iloc[:,index].dtype)
        cond1_for_num = 'float' in datatype or 'int' in datatype
        cond2_for_num = len(np.unique(df.iloc[:,index].values)) > n_unique_value_threshold
        if cond1_for_num and cond2_for_num:
            feature_type = 'numerical'
        else:
            feature_type = 'categorical'
        feature_types.append(feature_type)
    return feature_types


def get_potential_splits(data,feature_types):
    '''Return a dictionary of potential splits with keys 
        corresponding to columns of feature matrix and 
        values corresponding to pontial spilts in each 
        column''' 
    X = data[:,:-1] #Extracting the feature matrix
    potential_splits = {}
    for column_index in range(X.shape[1]):
        unique_values = np.unique(X[:,column_index])
        if feature_types[column_index] == 'numerical' and len(unique_values) > 1:
            potential_splits[column_index] = []
            for index in range(1,len(unique_values)): #Skiping the first index
                current_value = unique_values[index]
                previous_value = unique_values[index - 1]
                potential_split = (current_value + previous_value) / 2  
                potential_splits[column_index].append(potential_split)
        elif feature_types[column_index] == 'categorical' and len(unique_values) > 1:
            potential_splits[column_index] = list(unique_values)
    return potential_splits

def split_data(data,split_col,split_value,feature_types):
    feature_type = feature_types[split_col]
    
    if feature_type == 'numerical':
        data_left = data[data[:,split_col] <= split_value]
        data_right = data[data[:,split_col] > split_value ]

    elif feature_type == 'categorical':        
        data_left = data[data[:,split_col] == split_value]
        data_right = data[data[:,split_col] != split_value]
    
    return data_left, data_right

def leaf_mse(y):
    if len(y) == 0:
        mse = 0
    else:
        leaf_mean = np.mean(y)
        mse = np.mean((leaf_mean - y)**2)
    return mse

def split_mse(data_left,data_right):
    n = len(data_left) + len(data_right)
    p_left = len(data_left)/n
    p_right = len(data_right)/n
    split_mse = p_left*leaf_mse(data_left[:,-1]) + p_right*leaf_mse(data_right[:,-1])
    return split_mse

def determine_best_split(data,potential_splits,feature_types):
    first_iter = True #making sure that this if statement is executed atleast once, otherwise an arbitrarily high values must be set
    for column_index in potential_splits:
        for value in potential_splits[column_index]:
            data_left,data_right = split_data(data,column_index,value,feature_types)
            current_mse = split_mse(data_left,data_right)
            if first_iter or current_mse <= mse_best_split: #Although mse_best_split hasnt been created, it wont;
                first_iter = False                          #be a problem, bcz the first condition has been satised 
                mse_best_split = current_mse
                best_split_col = column_index
                best_split_value = value
    return best_split_col,best_split_value

def decision_tree_regression(data,min_samples,max_depth,column_headers,feature_types,counter):
    potential_splits = get_potential_splits(data,feature_types)
    #Base case, if the data is pure,
              # if there are no potential splits
              # if max depth is reached, classify the data
    if (len(data) < min_samples) or (counter == max_depth) or len(potential_splits) == 0:
        return create_leaf(data)                                                         # if there are no potential splits,
                                                                                      # classify the data
    #Recrusive Part, if base case is not satisfied
    else:
        counter += 1
        
        #helper functions
        split_col,split_val = determine_best_split(data,potential_splits,feature_types)
        data_left, data_right = split_data(data,split_col,split_val,feature_types) #f
        
        col_name = column_headers[split_col]
        feat_type = feature_types[split_col]
        
        if feat_type == 'numerical':
            question = f'{col_name} <= {split_val}'
        else: #feat_type is categorical
            question = f'{col_name} = {split_val}'
            
        subtree = {question: []}
        yes_ans = decision_tree_regression(data_left,min_samples,max_depth,column_headers,feature_types,counter)
        no_ans = decision_tree_regression(data_right,min_samples,max_depth,column_headers,feature_types,counter)
        
        #if after splitting, both dataset labels are same then instead of the subtree;
        #being a dict and creating further nodes, return any one of the answer and create a leaf
        #This has high revelence in a classification problem, not so much in regression
        if yes_ans == no_ans: 
            subtree = yes_ans
        else:
            subtree[question].append(yes_ans)
            subtree[question].append(no_ans)

    return subtree

def single_predict(example,tree):
    question = list(tree.keys())[0]
    col_name,operator,value = question.split()
    if operator == '<=':
        if str(example[col_name]) <= value:
            answer = tree[question][0]
        else:
            answer = tree[question][1]
    else:
         if str(example[col_name]) == value:
            answer = tree[question][0]
         else:
            answer = tree[question][1]
    if not isinstance(answer, dict):
        return answer
    else:
        residual_tree = answer
        return single_predict(example,residual_tree)