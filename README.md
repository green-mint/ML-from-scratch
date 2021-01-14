# ML-from-scratch
Popular Machine Learning Algorithms from Scratch using Python, Numpy.  
  
  
**NOTE:** This repository is only meant for understanding of under the hood working of machine learning algorithms. They are not optimized enough compared to sklearn or keras algorithms. Evaluations and comparisons with sklearn and keras will be provided.

## List of Alogrithms Implemented:  

* Multi-layered Perceptron - A basic Neural network
    * Layers:  
    Resembles the Keras architecture. A dense layer has been implemented with an additional parameter of keep_prob. This imitates the keras Dropout() layer. In future this will be separate layer just like in Keras.
    * Optimzers:
    Three optimizers including Adam, Momentum and Stochastic Gradient Descent can be used. RMS Prop and Adagrad will be added in future.

* Support Vector Machine:  
Kernel parameters are linear kernel, RBF kernel and the ploynomial kernel. Uses the [cvxopt](https://cvxopt.org/) package for the quadratic optimization of the loss function.

* Decision Tree Regressor and Classifier:  
Min_depth and Min_smaples parameters for reducing bias and overfitting. Random forest regressor and classifier can also be trained by using n_estimator number of decision trees.

* K-means:  
An unsupervised learning algorithm that groups related data based on the number of clusters.

* Convolution Neural Network:  
    TODO
    
## Evaluation:
TODO


## Directory Structure:

### deep_learning/

* [demo/](./deep_learning/demo)
  * [data/](./deep_learning/demo/data)
  * [demo.ipynb](./deep_learning/demo/demo.ipynb)
  * [test..py](./deep_learning/demo/test..py)
* [layer](./deep_learning/layer.py)
* [model](./deep_learning/model.py)
* [optimizers](./deep_learning/optimizers.py)

### supervised_learning/

* [data/](./supervised_learning/data)
* [decision-tree/](./supervised_learning/decision-tree)
  * [demo/](./supervised_learning/decision-tree/demo)
    * [dt_clasification_test.ipynb](./supervised_learning/decision-tree/demo/dt_clasification_test.ipynb)
    * [dt_regression_test.ipynb](./supervised_learning/decision-tree/demo/dt_regression_test.ipynb)
  * [DecisionTreeClassifier.py](./supervised_learning/decision-tree/DecisionTreeClassifier.py)
  * [DecisionTreeRegressor.py](./supervised_learning/decision-tree/DecisionTreeRegressor.py)
  * [random_forest.ipynb](./supervised_learning/decision-tree/random_forest.ipynb)
  * [utils_classif.py](./supervised_learning/decision-tree/utils_classif.py)
  * [utils_regres.py](./supervised_learning/decision-tree/utils_regres.py)
* [svm/](./supervised_learning/svm)
  * [SVM.py](./supervised_learning/svm/SVM.py)
  * [demo.ipynb](./supervised_learning/svm/demo.ipynb)

### unsupervised_learning/
* [k-means/](./unsupervised_learning/k-means)
  * [k_means.py](./unsupervised_learning/k-means/k_means.py)



