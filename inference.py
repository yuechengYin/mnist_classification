# Inference.

from utils.predictor import *
# Load the trained model parameters.
train_params = np.load('model.npy').item()
# Create the model.
inference_model = Predictor(train_params)
# Inference.
acc = inference_model.predict(X_test, Y_test)

print('The Final Test Accuracy:',acc)
