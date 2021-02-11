# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 15:07:46 2021

@author: Antonin
"""

import numpy as np

x = [1, 5]

x = np.array(x)

np.zeros([4, 5])

np.ones([4, 6,])

np.random.random([3, 2])

np.random.random([2, 1])



def randomization(n):
    """
    Arg:
      n - an integer
    Returns:
      A - a randomly-generated nx1 Numpy array.
    """
    #Your code here
    A = np.random.random(n)
    return A

    raise NotImplementedError

randomization(3)
randomization(5.7)

y = [6, 7]

y = np.array(y)

np.matmul(x, y)

np.exp(x)

np.sin(x)
np.cos(x)
np.tanh(x)

def operations(h, w):
    """
    Takes two inputs, h and w, and makes two Numpy arrays A and B of size
    h x w, and returns A, B, and s, the sum of A and B.

    Arg:
      h - an integer describing the height of A and B
      w - an integer describing the width of A and B
    Returns (in this order):
      A - a randomly-generated h x w Numpy array.
      B - a randomly-generated h x w Numpy array.
      s - the sum of A and B.
    """
    
    A = np.random.random([h, w])
    B = np.random.random([h, w])
    s = A + B
    return A, B, s
    raise NotImplementedError

operations(5, 3)

x = np.array([4,2,9,-6,5,11,3])

x
#Out[37]: array([ 4,  2,  9, -6,  5, 11,  3])

np.max(x)
Out[38]: 11

np.min(x)
Out[39]: -6

x.max()

#np.random.
#np.linalg.

def norm(A, B):
    """
    Takes two Numpy column arrays, A and B, and returns the L2 norm of their
    sum.

    Arg:
      A - a Numpy array
      B - a Numpy array
    Returns:
      s - the L2 norm of A+B.
    """
    
    s = np.linalg.norm(A+B)
    return s
    raise NotImplementedError

def neural_network(inputs, weights):
    """
     Takes an input vector and runs it through a 1-layer neural network
     with a given weight matrix and returns the output.

     Arg:
       inputs - 2 x 1 NumPy array
       weights - 2 x 1 NumPy array
     Returns (in this order):
       out - a 1 x 1 NumPy array, representing the output of the neural network
    """
    I = np.array(inputs)
    W = np.array(weights)
    
    Net_transform = np.tanh(np.matmul(I, W.transpose()))
    return np.array(Net_transform)
    raise NotImplementedError

neural_network([1,2], [0.01, 0.005])

def scalar_function(x, y):
    """
    Returns the f(x,y) defined in the problem statement.
    """
    if x <= y:
        return x * y
    else:
        return x/y
        
    raise NotImplementedError




def vector_function(x, y):
    """
    Make sure vector_function can deal with vector input x,y 
    """
    
    x = np.array(x)
    y = np.array(y)
    
    if np.any(x <= y):
        return x * y
    
    else:
        return x/y
    
    vfunc = np.vectorize(x, y)
    return np.array(vfunc)

    raise NotImplementedError

vector_function(([5, 2]), ([2,1]))