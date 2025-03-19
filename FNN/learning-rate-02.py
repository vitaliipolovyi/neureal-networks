# Drop-Based Learning Rate Decay
# %%
from pandas import read_csv
import math
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import LearningRateScheduler

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
sgd = SGD(learning_rate=0.1, nesterov=False)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
model.fit(X, Y, validation_split=0.33, epochs=50, batch_size=28, verbose=2)

# learning rate schedule
# %%
def step_decay(epoch):
	initial_lrate = 0.1
	drop = 0.5
	epochs_drop = 10.0
	lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
	return lrate

# %%
sgd = SGD(learning_rate=0.0, momentum=0.9)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
lrate = LearningRateScheduler(step_decay)
callbacks_list = [lrate]
model.fit(X, Y, validation_split=0.33, epochs=50, batch_size=28, callbacks=callbacks_list, verbose=2)

