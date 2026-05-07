import numpy as np

def softmax(x):

 #   x: (..., seq_len)

    x_shifted = x - np.max(x, axis=-1, keepdims=True)
    
    exp_x = np.exp(x_shifted)
    
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def softmax_backward(d_out, softmax_out):
   
   #   d_out: gradient wrt softmax output
   #  softmax_out: output of softmax
   #  shape: (..., seq_len)
   
    temp = np.sum(d_out * softmax_out, axis=-1, keepdims=True)

    d_x = softmax_out * (d_out - temp)

    return d_x