import numpy as np
import autograd.numpy as np_auto
from matplotlib import pyplot as plt

from TreeClassifier import question_4, question_5
from LinearModels import question_9, question_10
from NeuralNetwork import prediction_loss, prediction_grad
from AutoGrad import prediction_grad_autograd, question_17, question_18
from knnwithcrossval import question_7, question_8

stuff=np.load("../report_src/data.npz")
X_trn = stuff["X_trn"]
y_trn = stuff["y_trn"]
X_tst = stuff["X_tst"]

####Q:4 Train 6 different classification trees on the image data, with each of the following maximum depths
####{1,3,6,9,12,14}
####use 5-fold cross validation
# question_4(X_trn, y_trn, X_tst)

####Q:5 Use best depth in above question to predict results for test data and uplaod to kaggle
####best depth above is 6
# question_5(X_trn, y_trn, X_tst)

####Q:7 Do nearest-neighbor prediction for each of the following possible values of K: {1, 3, 5, 7, 9, 11}. 
### Using 5-fold cross-validation, estimate the out of sample classification error, and report this as a table.
### refer knnwithcrossval.py
# question_7(X_trn, y_trn, X_tst) 

####Q:8 What K performs best in the previous question? Using that K, make predictions on the test data
# question_8(X_trn, y_trn, X_tst)

###Q:9 For both hinge loss and logistic loss, train linear models with ridge regularization
###train the model and estimate the mean out of sample loss/error using 5-fold cross-validation
###l = {.0001,.001,1,10,100}
# question_9(X_trn, y_trn, X_tst)

### Q:10 Choose the training loss and \lambda that you think will perform best
### Make predictions for the test data
###best lambda above: , best training loss estimator: 
# question_10(X_trn, y_trn, X_tst)

#### Q:11 Write a function to evaluate the neural network and loss
###prediction_loss(x,y,W,V,b,c) function in NeuralNetwork.py

#### Q:12 Write a function to evaluate the gradient of the neural network
###prediction_grad(x,y,W,V,b,c) function in NeuralNetwork.py

# #### Q:13 Compute gradient wrt W,v,b,c using q:12
# x = np.array([1, 2], dtype="float")
# y=1
# W=np.array([[0.5,-1], [-0.5, 1], [1, 0.5]], dtype="float")
# V=np.array([[-1,-1, 1], [1, 1, 1]], dtype="float")
# b=np.array([0, 0, 0], dtype="float")
# c=np.array([0, 0], dtype="float")
# dLdW, dLdV, dLdb, dLdc = prediction_grad(x,y,W,V,b,c)
# print(f"dLdW, dLdV, dLdb, dLdc: {dLdW}, {dLdV}, {dLdb}, {dLdc}")

# #### Q:14 Write a function to evaluate the same gradient as in Question 12 using the autograd toolbox
# x = np_auto.array([1, 2], dtype="float")
# y=1
# W=np_auto.array([[0.5,-1], [-0.5, 1], [1, 0.5]], dtype="float")
# V=np_auto.array([[-1,-1, 1], [1, 1, 1]], dtype="float")
# b=np_auto.array([0, 0, 0], dtype="float")
# c=np_auto.array([0, 0], dtype="float")
# dLdW, dLdV, dLdb, dLdc = prediction_grad_autograd(x,y,W,V,b,c)
# print("i used autograd:")
# print(f"dLdW, dLdV, dLdb, dLdc: {dLdW}, {dLdV}, {dLdb}, {dLdc}")

##### Q:15 Update your gradient function to work on a full dataset and include regularization, as in the previous question
###prediction_loss_full(x,y,W,V,b,c)

##### Q:16 Update your gradient function to work on a full dataset and include regularization, as in the previous question. 
###prediction_grad_full(X,Y,W,V,b,c,lam)

##### Q:17 Optimize a function h(w) by gradient descent with momentum.
# question_17(np_auto.array(X_trn), y_trn)

##### Q:18 Train NN using train validation split of 50-50
# question_18(np_auto.array(X_trn), y_trn)

