from builtins import range
import numpy as np


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None

    N = x.shape[0]
    out = x.reshape(N,-1).dot(w)+b
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None

    N = x.shape[0]
    dx = dout.dot(w.T).reshape(x.shape)
    dw = x.reshape(N,-1).T.dot(dout)
    db = np.sum(dout,axis=0)
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    out = np.maximum(0, x)
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    dx = dout
    dx[x<=0] = 0
    return dx



def conv_forward(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width HH.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None

    # Get the pad and stride
    pad = conv_param['pad']
    stride = conv_param['stride']

    # Get dimensions
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    H_filter = 1+(H+2*pad-HH)//stride
    W_filter = 1+(W+2*pad-WW)//stride

    # Initialize output matrix with zeros
    out = np.zeros((N, F, H_filter, W_filter))

    # Specify padding location
    npad = ((0,0), (0,0), (pad, pad), (pad, pad))

    # Pad the input with zeros
    x = np.pad(x, pad_width=npad, mode='constant', constant_values=0)

    # Translate filters across the input
    for i in range(N):
      for z in range(F):
          for j in range(H_filter):
              for k in range(W_filter):
                  out[i, z, j, k] = np.sum(x[i,:,j*stride:(j*stride+HH),k*stride:(k*stride+WW)]*w[z,:,:,:])+b[z]


    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward(dout, cache):
    """
    Implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """



    dx, dw, db = None, None, None

    x, w, b, conv_param = cache
  
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    stride, pad = conv_param['stride'], conv_param['pad']
    H_out = 1 + (H + 2 * pad - HH) // stride
    W_out = 1 + (W + 2 * pad - WW) // stride
  
    x_pad = np.pad(x, ((0,), (0,), (pad,), (pad,)), mode='constant', constant_values=0)
    dx = np.zeros_like(x)
    dx_pad = np.zeros_like(x_pad)
    dw = np.zeros_like(w)
    db = np.zeros_like(b)
  
    db = np.sum(dout, axis = (0,2,3))
  
    x_pad = np.pad(x, ((0,), (0,), (pad,), (pad,)), mode='constant', constant_values=0)
    for i in range(H_out):
        for j in range(W_out):
            x_pad_masked = x_pad[:, :, i*stride:i*stride+HH, j*stride:j*stride+WW]
            for k in range(F): #compute dw
                dw[k ,: ,: ,:] += np.sum(x_pad_masked * (dout[:, k, i, j])[:, None, None, None], axis=0)
            for n in range(N): #compute dx_pad
                dx_pad[n, :, i*stride:i*stride+HH, j*stride:j*stride+WW] += np.sum((w[:, :, :, :] * 
                                                 (dout[n, :, i, j])[:,None ,None, None]), axis=0)
    dx = dx_pad

    return dx, dw, db




def max_pool_forward(x, pool_param):
    """
    A naive implementation of the forward pass for a max pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    Returns a tuple of:
    - out: Output data
    - cache: (x, pool_param)
    """
    out = None

    # Get dimensions
    N, C, H, W = x.shape
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']

    # Compute dimension filters
    H_filter = (H-pool_height)//stride+1
    W_filter = (W-pool_width)//stride+1

    # Initialize empty pooling layer
    out = np.zeros((N, C, H_filter, W_filter))

    # Translate filters across the input
    for j in range(H_filter):
      for k in range(W_filter):
          out[:,:,j,k] = (x[:,:,j*stride:(j*stride+pool_height),
             k*stride:(k*stride+pool_width)].max(axis=(2,3)))


    cache = (x, pool_param)
    return out, cache


def max_pool_backward(dout, cache):
    """
    Implementation of the backward pass for a max pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None


    # Unroll variables in cache
    x, pool_param = cache

    # Get dimensions
    N, C, H, W = x.shape
    HH = pool_param['pool_height']
    WW = pool_param['pool_width']
    stride = pool_param['stride']

    # Compute dimension filters
    H_filter = (H-HH) // stride+1
    W_filter = (W-WW) // stride+1

    # Initialize tensor for dx
    dx = np.zeros_like(x)

    # Backpropagate dout on x
    for i in range(N):
      for z in range(C):
          for j in range(H_filter):
              for k in range(W_filter):
                  dpatch = np.zeros((HH,WW))
                  input_patch = x[i,z,j*stride:(j*stride+HH),k*stride:(k*stride+WW)]
                  idxs_max = np.where(input_patch==input_patch.max())
                  dpatch[idxs_max[0], idxs_max[1]] = dout[i,z,j,k]
                  dx[i,z,j*stride:(j*stride+HH),k*stride:(k*stride+WW)] += dpatch


    return dx







def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx


