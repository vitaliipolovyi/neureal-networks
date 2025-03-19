# %%
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Layer, Input, Dense
from tensorflow.keras.models import Model

# %%
np.random.seed(0)
X = np.random.uniform(-1, 1, (100, 2))  # 100 samples, 2 features
y = X[:, 0]**2 + X[:, 1]**2 + np.random.normal(0, 0.1, 100)  # Quadratic function with noise

# %%
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# %%
class FuzzyLayer(Layer):
    def __init__(self, output_dim, **kwargs):
        super(FuzzyLayer, self).__init__(**kwargs)
        self.output_dim = output_dim

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)

    def call(self, x):
        return tf.math.sigmoid(tf.matmul(x, self.kernel))

# %%
input_layer = Input(shape=(2,))
fuzzy_layer = FuzzyLayer(10)(input_layer)
hidden_layer = Dense(10, activation='relu')(fuzzy_layer)
output_layer = Dense(1)(hidden_layer)

model = Model(inputs=input_layer, outputs=output_layer)
model.compile(optimizer='adam', loss='mean_squared_error')

# %%
model.fit(X_train, y_train, epochs=1000, batch_size=10, verbose=1)

# %%
predictions = model.predict(X_test)
mse = np.mean((y_test - predictions.flatten())**2)
print("Mean Squared Error:", mse)

# %%
plt.scatter(X_test[:,0], y_test, label='True')
plt.scatter(X_test[:,0], predictions, label='Predicted')
plt.legend()
plt.title("Fuzzy Neural Network Predictions")
plt.show()
