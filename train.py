import numpy as np
from utils.layers import *
from utils.solver import Solver
from utils.load_mnist import load_mnist_data
from utils.LeNet import *


# load the training data and validation data.
training_data, validation_data, test_data = load_mnist_data()
X_train, Y_train = training_data
X_val, Y_val = validation_data
total_data = {
  'X_train': X_train,
  'y_train': Y_train,
  'X_val': X_val,
  'y_val': Y_val,
}

# LeNet model implementation.
model = LeNet(weight_scale=1e-2)

solver = Solver(model, total_data,
                num_epochs=8, batch_size=256,
                update_rule='adam',
                optim_config={
                  'learning_rate': 1e-2,
                },
                verbose=True, print_every=1)
solver.train()
solver.save()
