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