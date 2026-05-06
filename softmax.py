import numpy as np

def softmax(x):

 #   x: (..., seq_len)

    x_shifted = x - np.max(x, axis=-1, keepdims=True)
    
    exp_x = np.exp(x_shifted)
    
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)