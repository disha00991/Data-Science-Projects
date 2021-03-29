import autograd, autograd.misc
import autograd.numpy as np
from autograd import grad
import math

def sigma(x):
    return np.tanh(x)

def f_x(x, y, W, V, b, c):
    return c + np.matmul(sigma(b + np.matmul(x, np.transpose(W))), np.transpose(V))

def L_y_f(x, y, W, V, b, c):
    g = f_x(x,y,W,V,b,c)
    j = np.sum(np.exp(g))
    print(j)
    h = np.log(j)
    print(g)
    print(h)
    return -1*g[y] + h

def prediction_grad_autograd(x, y, W, V, b, c):
    gradient = autograd.grad(L_y_f, [2])
    dLdW = gradient(x, y, W, V, b, c)
    gradient = autograd.grad(L_y_f, [3])
    dLdV = gradient(x, y, W, V, b, c)
    gradient = autograd.grad(L_y_f, [4])
    dLdb = gradient(x, y, W, V, b, c)
    gradient = autograd.grad(L_y_f, [5])
    dLdc = gradient(x, y, W, V, b, c)
    return dLdW, dLdV, dLdb, dLdc


