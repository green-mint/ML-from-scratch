import numpy as np 
import pandas as pd

#Check Data purity
def purity_check(data):
    y = data[:,-1]
    if len(set(y)) == 1:
        return True
    else:
        return False

def classify(data):
    '''Returns the most frequent label occuring in the
        dependent variable data'''
    y = data[:,-1]
    unique_labels,label_counts = np.unique(y,return_counts=True)
    classification = unique_labels[np.argmax(label_counts)]
    return classification

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

def get_potential_splits(data,feature_types,choose_k_lessthan_d=False):
    '''Return a dictionary of potential splits with keys 
        corresponding to columns of feature matrix and 
        values corresponding to pontial spilts in each 
        column''' 
    X = data[:,:-1] #Extracting the feature matrix
    dimension_indexes = np.array([x for x in range(X.shape[1])])
    k = int(np.round(X.shape[1] ** (1/2)))
    if choose_k_lessthan_d:
        indexes = np.random.choice(dimension_indexes,size=k,replace=False)
    else:
        indexes = dimension_indexes
    potential_splits = {}
    for column_index in indexes:
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

def leaf_entropy(y):
    _,counts = np.unique(y,return_counts=True)
    probabilities = counts/counts.sum()
    entropy = np.sum(probabilities*-np.log2(probabilities))
    return entropy

def split_entropy(data_left,data_right):
    n = len(data_left) + len(data_right)
    p_left = len(data_left)/n
    p_right = len(data_right)/n
    entropy = p_left*leaf_entropy(data_left[:,-1]) + p_right*leaf_entropy(data_right[:,-1])
    return entropy

def determine_best_split(data,potential_splits,feature_types):
    entropy_best_split = 999
    for column_index in potential_splits:
        for value in potential_splits[column_index]:
            data_left,data_right = split_data(data,column_index,value,feature_types)
            entropy = split_entropy(data_left,data_right)
            if entropy <= entropy_best_split:
                entropy_best_split = entropy
                best_split_col = column_index
                best_split_value = value
    return best_split_col,best_split_value

def decision_tree_classification(data,min_samples,max_depth,column_headers,feature_types,counter,choose_k_lessthan_d):
    potential_splits = get_potential_splits(data,feature_types,choose_k_lessthan_d)
    #Base case, if the data is pure
    if (purity_check(data)) or (len(data) < min_samples) or (counter == max_depth) or len(potential_splits) == 0:
        return classify(data)                                                         # if there are no potential splits,
                                                                                      # classify the data
    #Recrusive Part
    else:
        counter += 1
        #helper functions
        split_col,split_val = determine_best_split(data,potential_splits,feature_types)
        data_left, data_right = split_data(data,split_col,split_val,feature_types) #f
        
        col_name = column_headers[split_col]
        feat_type = feature_types[split_col]
        if feat_type == 'numerical':
            question = f'{col_name} <= {split_val}'
        else:
            question = f'{col_name} = {split_val}'
        subtree = {question: []}
        yes_ans = decision_tree_classification(data_left,min_samples,max_depth,column_headers,feature_types,counter,choose_k_lessthan_d)
        no_ans = decision_tree_classification(data_right,min_samples,max_depth,column_headers,feature_types,counter,choose_k_lessthan_d)
        
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