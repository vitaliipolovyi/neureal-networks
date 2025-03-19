# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# %%
def sigmoid(x, derivative=False):
    if (derivative == True):
        return sigmoid(x, derivative=False) * (1 - sigmoid(x, derivative=False))
    else:
        return 1 / (1 + np.exp(-x))

def mean_squared_error(y_pred, y_true):
    return ((y_pred - y_true)**2).sum() / (2*y_pred.size)

def accuracy(y_pred, y_true):
    acc = np.absolute(np.array(y_pred) - np.array(y_true))
    return acc.mean()

# %%
np.random.seed(1)
eta = .6
num_hidden = 3

# %%
X = np.array([
    [0, 0, 1],
    [0, 1, 1],
    [1, 0, 0],
    [1, 1, 0],
    [1, 0, 1],
    [1, 1, 1],
])

# %%
y = np.array([[0, 1, 0, 1, 1, 0]]).T

# %%
hidden_weights = 2*np.random.random((X.shape[1] + 1, num_hidden)) - 1
output_weights = 2*np.random.random((num_hidden + 1, y.shape[1])) - 1

# %%
for epochs in [10, 100, 500, 1000, 2000, 5000, 10000, 20000]:
    print("Epochs: {}".format(epochs))
    results = pd.DataFrame(columns=["mse", "acc"])
    for i in range(epochs):
        # 1)
        input_layer_outputs = np.hstack((np.ones((X.shape[0], 1)), X))
        hidden_layer_outputs = np.hstack((np.ones((X.shape[0], 1)), sigmoid(np.dot(input_layer_outputs, hidden_weights))))
        output_layer_outputs = np.dot(hidden_layer_outputs, output_weights)
        # 2)
        output_error = output_layer_outputs - y
        hidden_error = hidden_layer_outputs[:, 1:] * (1 - hidden_layer_outputs[:, 1:]) * np.dot(output_error, output_weights.T[:, 1:])

        mse = mean_squared_error(output_layer_outputs, y)
        acc = accuracy(output_layer_outputs, y)
        results.loc[len(results)] = {"mse": mse, "acc": acc}

        # 3)
        hidden_pd = input_layer_outputs[:, :, np.newaxis] * hidden_error[: , np.newaxis, :]
        output_pd = hidden_layer_outputs[:, :, np.newaxis] * output_error[:, np.newaxis, :]

        total_hidden_gradient = np.average(hidden_pd, axis=0)
        total_output_gradient = np.average(output_pd, axis=0)

        hidden_weights += - eta * total_hidden_gradient
        output_weights += - eta * total_output_gradient

    print("Output: \n{}".format(output_layer_outputs))
    results.mse.plot(title="Mean Squared Error")
    plt.show()
    #results.acc.plot(title="Accuracy")
    #plt.show()
    print("----------\n")

