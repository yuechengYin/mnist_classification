import numpy as np

from utils.layers import *
from utils.layer_utils import *


class LeNet(object):
  """
  Implementation of LeNet only using numpy.
  """
  
  def __init__(self, input_dim=(1, 28, 28), num_classes=10, weight_scale=1e-3, reg=0.0, dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    

    C, H, W = input_dim
    self.params['W1'] = weight_scale * np.random.randn(6, C, 5, 5) # 1st layer: conv filter.
    self.params['b1'] = np.zeros(6)
    self.params['W2'] = weight_scale * np.random.randn(16, 6, 5, 5) # 2ne layer: conv filter.
    self.params['b2'] = np.zeros(16)
    self.params['W3'] = weight_scale * np.random.randn(400, 120)   # 3rd layer: fc weights
    self.params['b3'] = np.zeros(120)
    self.params['W4'] = weight_scale * np.random.randn(120, 84)    # 4th layer: fc weights
    self.params['b4'] = np.zeros(84)
    self.params['W5'] = weight_scale * np.random.randn(84, 10)     # 5th layer: fc weights
    self.params['b5'] = np.zeros(10)

    # change the data type.
    for k, v in self.params.items():
      self.params[k] = v.astype(dtype)
     
  def padInput(self, X):
    '''
    Pad the origin image in mnist, from 28*28 to 32*32.
    '''
    pad = 2
    # Specify padding location
    npad = ((0,0), (0,0), (pad, pad), (pad, pad))

    # Pad the input with zeros
    X_pad = np.pad(X, pad_width=npad, mode='constant', constant_values=0)
    return X_pad

  def loss(self, X, y=None, params=None):
    """
    Evaluate loss and gradient for the LeNet.
    
    """

    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    W4, b4 = self.params['W4'], self.params['b4']
    W5, b5 = self.params['W5'], self.params['b5']
    
    # pass conv_param to the forward pass for the convolutional layer

    conv_param = {'stride': 1, 'pad': 0}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None

    X = self.padInput(X)
    # conv1 + relu1 + max_pool1
    conv_forward_out_1, cache_forward_1 = conv_relu_pool_forward(X, self.params['W1'], self.params['b1'], conv_param, pool_param)
    # conv2 + relu2 + max_pool2
    conv_forward_out_2, cache_forward_2 = conv_relu_pool_forward(conv_forward_out_1, self.params['W2'], self.params['b2'], 
                                                                  conv_param,     pool_param)
    # fc1 + relu
    affine_forward_out_3, cache_forward_3 = affine_forward(conv_forward_out_2, self.params['W3'], self.params['b3'])
    affine_relu_3, cache_relu_3 = relu_forward(affine_forward_out_3)
    # fc2 + relu
    affine_forward_out_4, cache_forward_4 = affine_forward(affine_relu_3, self.params['W4'], self.params['b4'])
    affine_relu_4, cache_relu_4 = relu_forward(affine_forward_out_4)

    # compute cls scores
    scores, cache_forward_5 = affine_forward(affine_relu_4, self.params['W5'], self.params['b5'])
    
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    loss, dout = softmax_loss(scores, y)

    # Add regularization
    l_aid= np.sum(self.params['W1'] ** 2) + np.sum(self.params['W2'] ** 2) + np.sum(self.params['W3'] ** 2)
    l_aid += np.sum(self.params['W4'] ** 2) + np.sum(self.params['W5'] ** 2)
    loss += self.reg * 0.5 * l_aid

    dX5, grads['W5'], grads['b5'] = affine_backward(dout, cache_forward_5)

    dX4 = relu_backward(dX5, cache_relu_4)
    dX4, grads['W4'], grads['b4'] = affine_backward(dX4, cache_forward_4)

    dX3 = relu_backward(dX4, cache_relu_3)
    dX3, grads['W3'], grads['b3'] = affine_backward(dX3, cache_forward_3)

    dX2, grads['W2'], grads['b2'] = conv_relu_pool_backward(dX3, cache_forward_2)


    dX1, grads['W1'], grads['b1'] = conv_relu_pool_backward(dX2, cache_forward_1)

    grads['W5'] = grads['W5'] + self.reg * self.params['W5']
    grads['W4'] = grads['W4'] + self.reg * self.params['W4']
    grads['W3'] = grads['W3'] + self.reg * self.params['W3']
    grads['W2'] = grads['W2'] + self.reg * self.params['W2']
    grads['W1'] = grads['W1'] + self.reg * self.params['W1']
    
    return loss, grads
  
