# %%
import numpy as np
from utils import cross_entropy_loss, cross_entropy_loss_gradient
from layers.Conv2D import Conv2D
from layers.MaxPool2D import MaxPool2D
from layers.Dense import Dense

# %%
def train_network(X, y, layers, lr=0.01, epochs=200):
    x_range = range(len(X))

    for epoch in range(epochs):
        total_loss = 0.0
        correct_predictions = 0
        
        for i in x_range:
            # Forward pass
            input = X[i]
            full_out = []
            
            #conv_out = layers[0].forward(X[i])
            #pool_out = layers[1].forward(conv_out)
            #full_out = layers[2].forward(pool_out)

            for layer in layers:
                full_out = layer.forward(input)
                input = full_out
                
            loss = cross_entropy_loss(full_out.flatten(), y[i])
            total_loss += loss

            # Converting to One-Hot encoding
            one_hot_pred = np.zeros_like(full_out)
            one_hot_pred[np.argmax(full_out)] = 1
            one_hot_pred = one_hot_pred.flatten()

            num_pred = np.argmax(one_hot_pred)
            num_y = np.argmax(y[i])

            if num_pred == num_y:
                correct_predictions += 1

            # Backward pass
            gradient = cross_entropy_loss_gradient(y[i], full_out.flatten()).reshape((-1, 1))
            #full_back = layers[2].backward(gradient, lr)
            #pool_back = layers[1].backward(full_back, lr)
            #conv_back = layers[0].backward(pool_back, lr)
            input_back = gradient
            for layer_reversed in reversed(layers):
                output_back = layer_reversed.backward(input_back, lr)
                input_back = output_back

        average_loss = total_loss / len(X)
        accuracy = correct_predictions / len(X_train) * 100.0
        
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {average_loss:.4f} - Accuracy: {accuracy:.2f}%")
        #break
        
def predict(input_sample, layers):
    input = input_sample
    for layer in layers[:-1]:    
        output = layer.forward(input)
        input = output

    flattened_output = output.flatten()
    
    predictions = layers[-1].forward(flattened_output)
    
    return predictions


# %%
import tensorflow as tf
import tensorflow.keras as keras

(train_images, train_labels), (test_images, test_labels) = keras.datasets.fashion_mnist.load_data()

# %%
train_images_raw = train_images[:5000]
X_train = train_images_raw / 255.0
y_train = train_labels[:5000]

X_test = train_images[5000:10000] / 255.0
y_test = train_labels[5000:10000]

y_labels_names = {
    0: 'T-shirt/top',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle boot',
}

# %%
from matplotlib import pyplot as plt

images_count_to_show = [4, 4]
fig, axs = plt.subplots(images_count_to_show[0], images_count_to_show[1], figsize=(128, 128))
img_it = 0
for img_it_x in range(images_count_to_show[0]):
    for img_it_y in range(images_count_to_show[1]):
        image_to_show = train_images_raw[img_it]
        axs[img_it_x, img_it_y].get_xaxis().set_visible(False)
        axs[img_it_x, img_it_y].get_yaxis().set_visible(False)
        axs[img_it_x, img_it_y].set_title("Label %s" % y_labels_names[y_train[img_it]], fontsize=100)
        axs[img_it_x, img_it_y].imshow(image_to_show / 255., cmap='gray')
        img_it = img_it + 1
plt.show()

# %%
from keras.utils import to_categorical

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(y_test[0])

# %%
conv = Conv2D(X_train[0].shape, 6, 1)
pool = MaxPool2D(2)
dense = Dense(121, 10)

layers = [
    conv,
    pool,
    dense
]

# %%
for x_train_it in range(len(X_train[:16])):
    fig, axs = plt.subplots(1, 3)

    axs[0].set_title('Original')
    axs[0].imshow(X_train[x_train_it] / 255., cmap='gray')
    
    conv_output = conv.forward(X_train[x_train_it])
    axs[1].set_title('Conv 1')
    axs[1].imshow(conv_output[0] / 255., cmap='gray')

    pool_output = pool.forward(conv_output)
    axs[2].set_title('Pool 1')
    axs[2].imshow(pool_output[0] / 255., cmap='gray')

    plt.show()

# %%
def train_network_wp(X, y, layers, lr=0.01, epochs=1):
    x_range = range(len(X))

    for epoch in range(epochs):
        total_loss = 0.0
        correct_predictions = 0
        
        for i in x_range:
            # Forward pass
            input = X[i]
            full_out = []
            
            for layer in layers[:-1]:
                full_out = layer.forward(input)
                input = full_out
                
            print(full_out.shape)
            plt.title(f'Epoch {epoch + 1} forward')
            plt.imshow(full_out[0] / 255., cmap='gray')
            plt.show()
            
            full_out = layers[-1].forward(input)
                
            loss = cross_entropy_loss(full_out.flatten(), y[i])
            total_loss += loss

            # Converting to One-Hot encoding
            one_hot_pred = np.zeros_like(full_out)
            one_hot_pred[np.argmax(full_out)] = 1
            one_hot_pred = one_hot_pred.flatten()

            num_pred = np.argmax(one_hot_pred)
            num_y = np.argmax(y[i])

            if num_pred == num_y:
                correct_predictions += 1

            # Backward pass
            gradient = cross_entropy_loss_gradient(y[i], full_out.flatten()).reshape((-1, 1))

            input_back = gradient
            for layer_reversed in reversed(layers):
                output_back = layer_reversed.backward(input_back, lr)
                input_back = output_back
                
        
        plt.title(f'Epoch {epoch + 1} back')
        plt.imshow(output_back / 255., cmap='gray')
        plt.show()

        average_loss = total_loss / len(X)
        accuracy = correct_predictions / len(X_train) * 100.0
        
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {average_loss:.4f} - Accuracy: {accuracy:.2f}%")

print(X_train[0].shape)
plt.title('Original')
plt.imshow(X_train[0] / 255., cmap='gray')
plt.show()

train_network_wp([X_train[0]], [y_train[0]], layers, 2)



# for x_train_it in range(len(X_train[:16])):
#     fig, axs = plt.subplots(1, 3)

#     axs[0].set_title('Original')
#     axs[0].imshow(X_train[x_train_it] / 255.)
    
#     conv_output = conv.forward(X_train[x_train_it])
#     axs[1].set_title('Conv 1')
#     axs[1].imshow(conv_output[0] / 255.)

#     pool_output = pool.forward(conv_output)
#     axs[2].set_title('Pool 1')
#     axs[2].imshow(pool_output[0] / 255.)

#     plt.show()
# %%
