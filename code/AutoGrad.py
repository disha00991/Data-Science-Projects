import autograd, autograd.misc
import autograd.numpy as np
from autograd import grad
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import time
import math

def sigma(x):
    return np.tanh(x)

def f_x(x, y, W, V, b, c):
    return c + np.matmul(sigma(b + np.matmul(x, np.transpose(W))), np.transpose(V))

def L_y_f(x, y, W, V, b, c):
    g = f_x(x,y,W,V,b,c)
    h = np.log(np.sum(np.exp(g)))
    return np.add(-1*g[y], h)

def prediction_grad_autograd(x, y, W, V, b, c):
    gradient = autograd.grad(L_y_f, [2, 3, 4, 5])
    dLdW, dLdV, dLdb, dLdc = gradient(x, y, W, V, b, c)
    return dLdW, dLdV, dLdb, dLdc

def prediction_loss_full(X,Y,W,V,b,c,lam):
    ##initialize Loss with regularization term
    L = lam*(np.sum(W**2) + np.sum(V**2))
    g = f_x(X,Y,W,V,b,c)
    h = np.log(np.sum(np.exp(g), 1))
    L += np.sum(-1*g[np.arange(len(Y)), Y] + h)
    return L 

def prediction_grad_full(X,Y,W,V,b,c,lam):
    gradient = autograd.grad(prediction_loss_full, [2, 3, 4, 5])
    dLdW, dLdV, dLdb, dLdc = gradient(X, Y, W, V, b, c, lam)
    return dLdW, dLdV, dLdb, dLdc

def question_17(X_trn, y_trn):
    print("M     |    Time taken")
    losses=[]
    M_array = [5, 40, 70]
    for M in M_array:
        start_time = time.time()
        W,V,b,c,loss = train_neural_network(X_trn,y_trn,M)
        losses.append(loss)
        end_time = time.time()
        print(f"    {M}    |    {(end_time - start_time)*1000}s")

    plot_loss(losses)

def question_18(X_trn, y_trn):
    kf = KFold(n_splits=2)
    for train_index, test_index in kf.split(X_trn):
      X_train, X_val = X_trn[train_index], X_trn[test_index]
      y_train, y_val = y_trn[train_index], y_trn[test_index]
      M_array = [5, 40, 70]
      errors = []
      print("   M    |   Error")
      for M in M_array:
          W,V,b,c,loss = train_neural_network(X_train,y_train.astype(int),M)
          y_pred = np.argmax(f_x(X_val,y_val,W,V,b,c), axis=1)
          errors.append(1 - accuracy_score(y_val,y_pred))

      chosenM = M_array[np.argmin(errors)]
      print(f"Chosen M: {chosenM}")
      print(f"Estimated Generalization Error: {np.mean(errors)}")
      print(f"   {M}   |    {error[i]}")
      break  #run the code only for first split

    #retrain on all data
    W,V,b,c,loss = train_neural_network(X_trn,y_trn,chosenM)
    y_pred = np.argmax(f_x(X_val,y_val,W,V,b,c), axis=1)    
    write_csv(y_pred, 'neuralnetwork.csv')

def plot_loss(loss):    
    r = list(range(1, 1001)) 
    plt.plot(r, loss[0], 'r-')
    plt.plot(r, loss[1], 'g-')
    plt.plot(r, loss[2], 'b-')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend([ 'M=5', 'M=40', 'M=70' ])
    plt.title('Regularized loss as a function of iterations');
    plt.show()

def train_neural_network(X_trn,y_trn,M):    
    D = len(X_trn[0])
    W = np.random.normal(loc=0.0, scale=1.0, size=[M,D])/np.sqrt(D)
    V = np.random.normal(loc=0.0, scale=1.0, size=[4,M])//np.sqrt(M)
    b = np.zeros(M) #dimension of M
    c = np.zeros(4) #dimension of no of classes in output which is 4 for our data y: {0,1,2,3}
    momentum = 0.1
    stepsize = 0.0001
    lam = 1 #regularization lambda 
    loss_points=[]
    ave_grad_w, ave_grad_v, ave_grad_b, ave_grad_c = [0,0,0,0]
    for iter in range(0, 1000):
        #For Loss function h,∇h(w) = dhdW which we calculate from above function
        dhdW, dhdV, dhdb, dhdc = prediction_grad_full(X_trn,y_trn,W,V,b,c,lam)
        #w
        ave_grad_w = (1 - momentum) * ave_grad_w + momentum * dhdW
        W = W - stepsize * ave_grad_w

        #v
        ave_grad_v = (1 - momentum) * ave_grad_v + momentum * dhdV
        V = V - stepsize * ave_grad_v

        #b
        ave_grad_b = (1 - momentum) * ave_grad_b + momentum * dhdb
        b = b - stepsize * ave_grad_b

        #c
        ave_grad_c = (1 - momentum) * ave_grad_c + momentum * dhdc
        c = c - stepsize * ave_grad_c

        loss_points.append(prediction_loss_full(X_trn,y_trn,W,V,b,c,lam))
    
    return W,V,b,c,loss_points

def write_csv(y_pred, filename):
    """Write a 1d numpy array to a Kaggle-compatible .csv file"""
    with open(filename, 'w') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Id', 'Category'])
        for idx, y in enumerate(y_pred):
            csv_writer.writerow([idx, y])      

