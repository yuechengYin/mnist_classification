from utils.LeNet import *
import numpy as np
from utils.layers import *
from utils.layer_utils import *


class Predictor(object):
    '''
    Predictor used for inference.
    '''
    def __init__(self, params):
    
        self.model_params = params
        self.model = LeNet()

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

    def compute_score(self, X)
        '''
        Compute the score.
        '''
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
		
		return scores

    def predict(self, X_test, Y_test):
        '''
        Predict the test data, return the accuracy.
        ''' 
        scores = self.compute_score(X_test)
        y_pred = np.argmax(scores, axis=1)
        y_pred = np.hstack(y_pred)
        acc = np.mean(y_pred == Y_test)

        return acc
