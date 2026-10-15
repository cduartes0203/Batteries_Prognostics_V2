
import numpy as np


def ResizeSignal(data, m):
    """
    Interpola um array de tamanho n para o tamanho m (m > n).
    """
    n = len(data)
    x_old = np.linspace(0, 1, n)
    x_new = np.linspace(0, 1, m)
    
    return np.interp(x_new, x_old, data)

def XavierUniform(shape,sd):
    np.random.seed(sd)
    n_in, n_out = shape
    limit = np.sqrt(6 / (n_in + n_out))
    return np.random.uniform(-limit, limit, size=shape) 

def Activation(x,mode='tanh'):
    '''mode: tanh, sigmoid, relu'''

    if mode == 'tanh':
        return np.tanh(x)
    elif mode == 'sigmoid':
        return 1 / (1 + np.exp(-x))
    elif mode == 'relu':
        return np.maximum(0, x)

def dActivation(x,mode='tanh'):
    '''mode: tanh, sigmoid, relu'''

    if mode == 'tanh':
        return 1 - np.tanh(x)**2
    elif mode == 'sigmoid':
        return (1 / (1 + np.exp(-x))) * (1- (1 / (1 + np.exp(-x))))
    elif mode == 'relu':
        return np.where(x > 0, 1, 0)