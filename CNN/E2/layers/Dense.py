import numpy as np

class Dense:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.randn(output_size, self.input_size)
        self.biases = np.random.rand(output_size, 1)
        
    def softmax(self, z):
        shifted_z = z - np.max(z)
        exp_values = np.exp(shifted_z)
        sum_exp_values = np.sum(exp_values, axis=0)

        probabilities = exp_values / sum_exp_values

        return probabilities
    
    def softmax_derivative(self, s):
        return np.diagflat(s) - np.dot(s, s.T)
    
    def forward(self, input_data):
        self.input_data = input_data
        # Зведення вхідних даних із попереднього шару у вектор
        flattened_input = input_data.flatten().reshape(1, -1)
        self.z = np.dot(self.weights, flattened_input.T) + self.biases

        self.output = self.softmax(self.z)

        return self.output
    
    def backward(self, dL_dout, lr):
        # Обчислення градієнта втрат відносно попередньої активації z
        dL_dy = np.dot(self.softmax_derivative(self.output), dL_dout)
        # Обчислення градієнт втрат відносно ваг dw
        dL_dw = np.dot(dL_dy, self.input_data.flatten().reshape(1, -1))
        # Обчислення градієнт втрат відносно зміщень (db)
        dL_db = dL_dy

        # Обчислення градієнт втрат відносно вхідних даних (dL_dinput)
        dL_dinput = np.dot(self.weights.T, dL_dy)
        dL_dinput = dL_dinput.reshape(self.input_data.shape)

        # Оновлення ваг та зміщення на основі швидкості навчання та градієнтів
        self.weights -= lr * dL_dw
        self.biases -= lr * dL_db

        # Повертаємо градієнт втрат відносто вхідних даних 
        return dL_dinput