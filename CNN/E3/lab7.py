# %%
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.metrics import Precision, Recall, F1Score
import cv2
import numpy as np
import matplotlib.pyplot as plt

# %%
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# %%
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# %%
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# %%
model_cnn = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

# %%
metrics = [
    'accuracy',
    Precision(),
    Recall(),
    F1Score(threshold=0.5),
]

# %%
model_cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=metrics)
history_cnn = model_cnn.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test), batch_size=64)

# %%
plt.plot(history_cnn.history['accuracy'])
plt.plot(history_cnn.history['val_accuracy'])
plt.title('Custom CNN model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history_cnn.history['loss'])
plt.plot(history_cnn.history['val_loss'])
plt.title('Custom CNN model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# %%
loss_cnn, *metrics_cnn = model_cnn.evaluate(x_test, y_test)

# %%
print("Accuracy", metrics_cnn[0])
print("Precision", metrics_cnn[1])
print("Recall", metrics_cnn[2])
print("F1Score", metrics_cnn[3])


# %%
base_model_vgg = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
base_model_resnet = ResNet50(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

# %%
def create_model(base_model):
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    predictions = Dense(10, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

# %%
model_vgg = create_model(base_model_vgg)
model_resnet = create_model(base_model_resnet)

# %%
model_vgg.compile(optimizer='adam', loss='categorical_crossentropy', metrics=metrics)
model_resnet.compile(optimizer='adam', loss='categorical_crossentropy', metrics=metrics)

# %%
history_vgg = model_vgg.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test), batch_size=64)
history_resnet = model_resnet.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test), batch_size=64)

# %%
plt.plot(history_vgg.history['accuracy'])
plt.plot(history_vgg.history['val_accuracy'])
plt.title('VGG model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history_vgg.history['loss'])
plt.plot(history_vgg.history['val_loss'])
plt.title('VGG model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# %%
plt.plot(history_resnet.history['accuracy'])
plt.plot(history_resnet.history['val_accuracy'])
plt.title('ResNet50 model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history_resnet.history['loss'])
plt.plot(history_resnet.history['val_loss'])
plt.title('ResNet50 model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# %%
loss_vgg, *metrics_vgg = model_vgg.evaluate(x_test, y_test)

# %%
print("Accuracy", metrics_vgg[0])
print("Precision", metrics_vgg[1])
print("Recall", metrics_vgg[2])
print("F1Score", metrics_vgg[3])

# %%
loss_resnet, *metrics_resnet = model_vgg.evaluate(x_test, y_test)

# %%
print("Accuracy", metrics_resnet[0])
print("Precision", metrics_resnet[1])
print("Recall", metrics_resnet[2])
print("F1Score", metrics_resnet[3])