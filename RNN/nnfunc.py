import numpy as np

def sigmoid(x, derivative=False):
  if (derivative == True):
    return sigmoid(x, derivative=False) * (1 - sigmoid(x, derivative=False))
  else:
    return 1 / (1 + np.exp(-x))

def tanh(x, derivative=False):
  if derivative == True:
    return 1.0 - x**2
  else:
    return np.tanh(x)

def softmax(x):
  return np.exp(x) / np.sum(np.exp(x), axis=0)

def rand_arr(l, r, ds1, ds2 = None):
  return np.random.uniform(l, r, (ds1, ds2) if ds2 else ds1)