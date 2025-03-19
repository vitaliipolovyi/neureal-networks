import numpy as np
import utils

class FullyConnected:                               # fully-connected layer
    def __init__(self, name, nodes1, nodes2, activation):
        self.name = name
        self.weights = np.random.randn(nodes1, nodes2) * 0.1
        self.biases = np.zeros(nodes2)
        self.activation = activation
        self.last_input_shape = None
        self.last_input = None
        self.last_output = None
        self.leaky_relu = np.vectorize(utils.leaky_relu)
        self.leaky_relu_derivative = np.vectorize(utils.leaky_relu_derivative)

    def forward(self, input):
        self.last_input_shape = input.shape         # keep track of last input shape before flattening
                                                    # for later backward propagation

        input = input.flatten()                                 # flatten input

        output = np.dot(input, self.weights) + self.biases      # forward propagate

        if self.activation == 'relu':                           # apply ReLU activation function
            self.leaky_relu(output)

        self.last_input = input                     # keep track of last input and output for later backward propagation
        self.last_output = output

        return output

    def backward(self, din, learning_rate=0.005):
        if self.activation == 'relu':                             # back propagate through ReLU
           self.leaky_relu_derivative(din)

        self.last_input = np.expand_dims(self.last_input, axis=1)
        din = np.expand_dims(din, axis=1)

        dw = np.dot(self.last_input, np.transpose(din))           # loss gradient of final dense layer weights
        db = np.sum(din, axis=1).reshape(self.biases.shape)       # loss gradient of final dense layer biases

        self.weights -= learning_rate * dw                        # update weights and biases
        self.biases -= learning_rate * db

        dout = np.dot(self.weights, din)

        return dout.reshape(self.last_input_shape)

    def get_weights(self):
        return np.reshape(self.weights, -1)