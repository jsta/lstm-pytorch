"""
Training the model at:
https://github.com/jessicayung/blog-code-snippets/blob/master/lstm-pytorch/lstm-baseline.py
"""
import torch
import pickle
import importlib
import numpy as np
import matplotlib.pyplot as plt

LSTM = importlib.import_module("99_lstm_class").LSTM

shim = ""  # "../"

#####################
# Set parameters
#####################

# Data params
test_size = 0.2
num_datapoints = 100

num_train = int((1 - test_size) * num_datapoints)
num_test = int(num_datapoints - num_train)

# Network params
input_size = 20

# If `per_element` is True, then LSTM reads in one timestep at a time.
per_element = True
if per_element:
    lstm_input_size = 1
else:
    lstm_input_size = input_size
# size of hidden layers
h1 = 32
output_dim = 1
num_layers = 2
learning_rate = 1e-3
num_epochs = 500
dtype = torch.float


# make training and test sets in torch
data = pickle.load(open(shim + "data/data.ardata", "rb"))

X_train = torch.from_numpy(data.X_train).type(torch.Tensor)
X_test = torch.from_numpy(data.X_test).type(torch.Tensor)
y_train = torch.from_numpy(data.y_train).type(torch.Tensor).view(-1)
y_test = torch.from_numpy(data.y_test).type(torch.Tensor).view(-1)

# 80 rows, 20 cols
# to
# 20 rows, 80 cols, 1 layer

X_train = X_train.view([input_size, -1, 1])
X_test = X_test.view([input_size, -1, 1])

#####################
# Build model
#####################

model = LSTM(
    lstm_input_size,
    h1,
    batch_size=num_train,
    output_dim=output_dim,
    num_layers=num_layers,
)

loss_fn = torch.nn.MSELoss(size_average=False)

optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)

#####################
# Train model
#####################

hist = np.zeros(num_epochs)

y_preds = []
for t in range(num_epochs):
    # Initialise hidden state
    # Don't do this if you want your LSTM to be stateful
    model.hidden = model.init_hidden()

    # Forward pass
    y_pred = model(X_train)
    y_preds.append(y_pred)

    loss = loss_fn(y_pred, y_train)
    if t % 100 == 0:
        print("Epoch ", t, "MSE: ", loss.item())
    hist[t] = loss.item()

    # Zero out gradient, else they will accumulate between epochs
    optimiser.zero_grad()

    # Backward pass
    loss.backward()

    # Update parameters
    optimiser.step()

# https://pytorch.org/tutorials/beginner/saving_loading_models.html
state = {"state_dict": model.state_dict(), "optimizer": optimiser.state_dict()}
torch.save(state, shim + "data/lstm-baseline_model.pytorch")
torch.save(y_preds, shim + "data/y_preds.pytorch")
torch.save(hist, shim + "data/hist.pytorch")
