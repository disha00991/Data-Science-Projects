import autograd, autograd.misc
import autograd.numpy as np
from autograd import grad
import math

def sigma(x):
    return np.tanh(x)

def f_x(x, y, W, V, b, c):
    return np.add(c, np.matmul(sigma(np.add(b, np.matmul(x, np.transpose(W)))), np.transpose(V)))

def L_y_f(x, y, W, V, b, c):
    g = f_x(x,y,W,V,b,c)
    h = np.log(np.sum(np.exp(g)))
    return np.add(-1*g[y], h)

def prediction_grad_autograd(x, y, W, V, b, c):
    gradient = autograd.grad(L_y_f, [2, 3, 4, 5])
    dLdW, dLdV, dLdb, dLdc = gradient(x, y, W, V, b, c)
    return dLdW, dLdV, dLdb, dLdc

def prediction_loss_full(X,Y,W,V,b,c,lam):
    ##Loss
    L = lam*(np.sum(W**2, V**2))
    for i, x in enumerate(X):
        L += L_y_f(x, Y[i], W, V, b, c)    
    return L

def prediction_grad_full(X,Y,W,V,b,c,lam):
    dLdW = [] 
    dLdV = []
    dLdb = []
    dLdc = []

    gW = autograd.grad(L_y_f, [2])
    gV = autograd.grad(L_y_f, [3])
    gb = autograd.grad(L_y_f, [4])
    gc = autograd.grad(L_y_f, [5])
    for i, x in enumerate(X):
        dLdW.append(gW(x, Y[i], W, V, b, c)+2*lam*(W))
        dLdV.append(gV(x, Y[i], W, V, b, c)+2*lam*(V))
        dLdb.append(gb(x, Y[i], W, V, b, c))
        dLdc.append(gc(x, Y[i], W, V, b, c))

    return dLdW, dLdV, dLdb, dLdc

def get_dhdW(X,Y,W,V,b,c,lam):
    dLdW = [] 
    gW = autograd.grad(L_y_f, [2])
    for i, x in enumerate(X):
        dLdW.append(gW(x, Y[i], W, V, b, c)+2*lam*(W))

    return dLdW

def question_17(X_trn, y_trn):
    M_array = [5, 40, 70]
    D = len(X_trn)
    W = 
    for M in M_array:
        V = 
        b = np.zeros(M) #dimension of M
        c = np.zeros(4) #dimension of no of classes in output which is 4 for our data y: {0,1,2,3}
        momentum = 0.1
        stepsize = 0.0001
        lam = 1 #regularization lambda 

        #For Loss function h,∇h(w) = dhdW which we calculate from above function
        dhdW = get_dhdW(X_trn,y_trn,W,V,b,c,lam)
        ave_grad = 0
        for iter in range(0, 1000):
            ave_grad = (1 - momentum) * ave_grad + momentum * dhdW
            W = W - stepsize * ave_grad


        