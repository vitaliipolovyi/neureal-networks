# Time-Based Learning Rate Schedule
# %%
from pandas import read_csv
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from keras.callbacks import Callback
from sklearn.preprocessing import LabelEncoder

# %%
class MyCallback(Callback):
  def __init__(self, initial_lr, decay = 0):
    super().__init__()
    self.initial_lr = initial_lr  # Store initial learning rate
    self.decay = decay  # Store decay rate

  def on_epoch_end(self, epoch, logs=None):
    optimizer = self.model.optimizer
    # Compute learning rate manually using SGD decay formula
    new_lr = self.initial_lr / (1 + self.decay * epoch)    
    # Get the momentum value
    momentum = tf.keras.backend.get_value(optimizer.momentum)
    print(f"Epoch {epoch+1}: Learning Rate = {new_lr}, Momentum = {momentum}")

# %%
dataframe = read_csv("ionosphere.csv", header=None)
dataset = dataframe.values
dataframe

# %%
X = dataset[:,0:34].astype(float)
Y = dataset[:,34]

# %%
encoder = LabelEncoder()
encoder.fit(Y)
Y = encoder.transform(Y)

# %%
model = Sequential()
model.add(Dense(34, input_shape=(34,), activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# %%
epochs = 50
learning_rate = 0.1

# %%
print_rl = MyCallback(learning_rate)
sgd = SGD(learning_rate=learning_rate, nesterov=False)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
model.fit(X, Y, validation_split=0.33, epochs=epochs, batch_size=28, verbose=2, callbacks=[print_rl])

# %%
decay_rate = learning_rate / epochs

# %%
print_rl = MyCallback(learning_rate, decay_rate)
sgd = SGD(learning_rate=learning_rate, decay=decay_rate, nesterov=False)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
model.fit(X, Y, validation_split=0.33, epochs=epochs, batch_size=28, verbose=2, callbacks=[print_rl])

# %%
momentum = 0.8

# %%
print_rl = MyCallback(learning_rate, decay_rate)
sgd = SGD(learning_rate=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
model.fit(X, Y, validation_split=0.33, epochs=epochs, batch_size=28, verbose=2, callbacks=[print_rl])
