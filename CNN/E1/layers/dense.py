import numpy as np
import utils

class Dense:                                        # dense layer with softmax activation
    def __init__(self, name, nodes, num_classes):
        self.name = name
        self.weights = np.random.randn(nodes, num_classes) * 0.1
        self.biases = np.zeros(num_classes)
        self.last_input_shape = None
        self.last_input = None
        self.last_output = None

    def forward(self, input):
        self.last_input_shape = input.shape         # keep track of last input shape before flattening
                                                    # for later backward propagation

        input = input.flatten()                                 # flatten input

        output = np.dot(input, self.weights) + self.biases      # forward propagate

        self.last_input = input                     # keep track of last input and output for later backward propagation
        self.last_output = output

        return utils.softmax(output)

    def backward(self, din, learning_rate=0.005):
        for i, gradient in enumerate(din):
            if gradient == 0:                   # the derivative of the loss with respect to the output is nonzero
                continue                        # only for the correct class, so skip if the gradient is zero

            t_exp = np.exp(self.last_output)                      # gradient of dout[i] with respect to output
            dout_dt = -t_exp[i] * t_exp / (np.sum(t_exp) ** 2)
            dout_dt[i] = t_exp[i] * (np.sum(t_exp) - t_exp[i]) / (np.sum(t_exp) ** 2)

            dt = gradient * dout_dt                               # gradient of loss with respect to output

            dout = self.weights @ dt                              # gradient of loss with respect to input

            # update weights and biases
            self.weights -= learning_rate * (np.transpose(self.last_input[np.newaxis]) @ dt[np.newaxis])
            self.biases -= learning_rate * dt

            return dout.reshape(self.last_input_shape)            # return the loss gradient for this layer's inputs

    def get_weights(self):
        return np.reshape(self.weights, -1)