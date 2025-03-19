import numpy as np
import utils

class Convolutional:
    def __init__(self, name, k_filters=16, stride=1, size=3, activation=None):
        self.name = name
        if isinstance(k_filters, int):
            self.filters = np.random.randn(k_filters, 3, 3) * 0.1
        else:
            self.filters = k_filters
        self.stride = stride
        self.size = size
        self.activation = activation
        self.last_input = None
        self.leaky_relu = np.vectorize(utils.leaky_relu)
        self.leaky_relu_derivative = np.vectorize(utils.leaky_relu_derivative)

    def forward(self, image):
        self.last_input = image                             # keep track of last input for later backward propagation

        input_dimension = image.shape[1]                                                # input dimension
        output_dimension = int((input_dimension - self.size) / self.stride) + 1         # output dimension

        out = np.zeros((self.filters.shape[0], output_dimension, output_dimension))     # create the matrix to hold the
                                                                                        # values of the convolution
        for f in range(self.filters.shape[0]):              # convolve each filter over the image,
            tmp_y = out_y = 0                               # moving it vertically first and then horizontally
            while tmp_y + self.size <= input_dimension:
                tmp_x = out_x = 0
                while tmp_x + self.size <= input_dimension:
                    patch = image[:, tmp_y:tmp_y + self.size, tmp_x:tmp_x + self.size]
                    out[f, out_y, out_x] += np.sum(self.filters[f] * patch)
                    tmp_x += self.stride
                    out_x += 1
                tmp_y += self.stride
                out_y += 1

        if self.activation == 'relu':                       # apply ReLU activation function
            self.leaky_relu(out)

        return out
    
    def backward(self, din, learn_rate=0.005):
        input_dimension = self.last_input.shape[1]          # input dimension

        if self.activation == 'relu':                       # back propagate through ReLU
           self.leaky_relu_derivative(din)

        dout = np.zeros(self.last_input.shape)              # loss gradient of the input to the convolution operation
        dfilt = np.zeros(self.filters.shape)                # loss gradient of filter

        for f in range(self.filters.shape[0]):              # loop through all filters
            tmp_y = out_y = 0
            while tmp_y + self.size <= input_dimension:
                tmp_x = out_x = 0
                while tmp_x + self.size <= input_dimension:
                    patch = self.last_input[:, tmp_y:tmp_y + self.size, tmp_x:tmp_x + self.size]
                    dfilt[f] += np.sum(din[f, out_y, out_x] * patch, axis=0)
                    dout[:, tmp_y:tmp_y + self.size, tmp_x:tmp_x + self.size] += din[f, out_y, out_x] * self.filters[f]
                    tmp_x += self.stride
                    out_x += 1
                tmp_y += self.stride
                out_y += 1

        self.filters -= learn_rate * dfilt # update filters using SGD

        return dout # return the loss gradient for this layer's inputs
    
    def get_weights(self):
        return np.reshape(self.filters, -1)