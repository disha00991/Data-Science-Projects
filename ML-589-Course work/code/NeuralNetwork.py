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
    f_x = c + np.matmul(V, sigma(b + np.matmul(W, x)))

    #first find dLdf = e_cap + exp(f_y)/summation(exp(f_i))
    dLdf = -1*unit_v(y, c) + np.exp(f_x)/np.sum(np.exp(f_x))

    #sigma(b+Wx)
    print("x", x.shape)
    print("w", W.shape)
    comp1 = sigma(b + np.matmul(W, x))
    #sigma V_t * dLdf
    comp2 = np.matmul(dLdf, V)
    #dLdW = comp1 * comp2 X x_t
    print("dldf", dLdf.shape)
    print("comp1", comp1.shape)
    dLdV = np.outer(dLdf, comp1)
    # dLdV = 0
    ## derivative of tanh = 1 - tanh^2
    dLdb = (1-comp1**2) * comp2
    dLdc = dLdf
    dLdW = np.outer(dLdb, x)

    return dLdW, dLdV, dLdb, dLdc

#returns unit vector
def unit_v(y, c):  
    unit_vector = np.zeros(len(c))
    unit_vector[y] = 1
    return  unit_vector