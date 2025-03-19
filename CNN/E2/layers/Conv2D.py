import numpy as np
from scipy.signal import correlate2d

class Conv2D:
    def __init__(self, input_shape, filter_size, num_filters):
        input_height, input_width = input_shape
        self.num_filters = num_filters
        self.input_shape = input_shape
        
        # Size of outputs and filters
        
        self.filter_shape = (num_filters, filter_size, filter_size) # (3,3)
        self.output_shape = (num_filters, input_height - filter_size + 1, input_width - filter_size + 1)
        
        self.filters = np.random.randn(*self.filter_shape)
        self.biases = np.random.randn(*self.output_shape)
        
    def forward(self, input_data):
        self.input_data = input_data
        
        # Згортка
        output = np.zeros(self.output_shape)
        for i in range(self.num_filters):
            output[i] = correlate2d(self.input_data, self.filters[i], mode='valid')
        
        # Relu активація
        output = np.maximum(output, 0)
        
        return output
    
    def backward(self, dL_dout, lr):
        # Генеруємо масив dL_dout випадковим чином
        dL_dinput = np.zeros_like(self.input_data)
        dL_dfilters = np.zeros_like(self.filters)

        for i in range(self.num_filters):
                # Розрахунок градієнта втрат щодо ядер
                dL_dfilters[i] = correlate2d(self.input_data, dL_dout[i],mode='valid')
                # Розрахунок градієнта втрат щодо входів
                dL_dinput += correlate2d(dL_dout[i],self.filters[i], mode='full')

        # Оновлення параметрів зі швидкістю навчання
        self.filters -= lr * dL_dfilters
        self.biases -= lr * dL_dout

        return dL_dinput