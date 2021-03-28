import numpy as np
import math

def sigma(x):
    return np.tanh(x)

def prediction_loss(x,y,W,V,b,c):
    #first find f(x) = c + V * sigma(b + Wx)
    f_x = c + np.matmul(sigma(b + np.matmul(x, np.transpose(W))), np.transpose(V))

    #second find loss L(y, f) = -f_y + log(summation(over all y) (e^f_y))
    L_y = -1*f_x[y] + math.log(np.sum(np.exp(f_x)))
    
    return L_y

def prediction_grad(x,y,W,V,b,c):
    f_x = c + np.matmul(sigma(b + np.matmul(x, np.transpose(W))), np.transpose(V))

    #first find dLdf = e_cap + exp(f_y)/summation(exp(f_i))
    dLdf = unit_v(f_x) + np.exp(f_x)/np.sum(np.exp(f_x))

    #sigma(b+Wx)
    print("x", x.shape)
    print("w", W.shape)
    comp1 = sigma(b + np.matmul(x, np.transpose(W)))
    #sigma V_t * dLdf
    comp2 = np.matmul(np.transpose(V), dLdf)
    #dLdW = comp1 * comp2 X x_t
    dLdW = np.outer(comp1 * comp2, np.transpose(x))
    print("dldf", dLdf.shape)
    print("comp1", comp1.shape)
    # dLdV = np.matmul(dLdf, np.transpose(comp1))
    dLdV = 0
    dLdb = comp1 * comp2
    dLdc = dLdf

    return dLdW, dLdV, dLdb, dLdc

#returns unit vector
def unit_v(v):
    vi = v/np.linalg.norm(v)
    print("length of unit v",sum(vi**2))
    return  v/np.linalg.norm(v)