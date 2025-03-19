# %%
from inout import load_mnist
from utils import preprocess, display_image
from network import Network
import skimage
from skimage import draw, io
from matplotlib import pyplot as plt
from layers.convolutional2 import Convolutional2
from layers.max_pooling import MaxPooling
import numpy as np

# %%
img = skimage.data.chelsea()
io.imshow(img)
plt.show()

# %%
grayscale_img = skimage.color.rgb2gray(img)
io.imshow(grayscale_img)
plt.show()
print(grayscale_img)
print(grayscale_img.shape)
grayscale_img = np.array([grayscale_img])

# %%
filters = np.array([
  [
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1],
  ],
  [
    [ 1, 1, 1],
    [ 0, 0, 1],
    [-1,-1,-1],
  ]
])
#print(grayscale_img)
#k_filters=3
conv1 = Convolutional2(name='conv1', k_filters=filters, stride=2, size=3, activation='relu')
conv1_imgs = conv1.forward(grayscale_img)

# %%
for conv1_img in conv1_imgs:
    print(conv1_img)
    io.imshow(conv1_img)
    plt.show()
# %%
mpool1 = MaxPooling(name='pool2', stride=2, size=2)
pool1_imgs = mpool1.forward(conv1_imgs)
for pool1_img in pool1_imgs:
    io.imshow(pool1_img)
    plt.show()

# %%
dataset_name = 'mnist'
num_epochs = 1 
learning_rate = 0.01
validate = 1
regularization = 0
verbose = 1
plot_weights = 1
plot_correct = 1
plot_missclassified = 1
plot_feature_maps = 1

print('\n--- Loading ' + dataset_name + ' dataset ---')                 # load dataset
dataset = load_mnist() # if dataset_name is 'mnist' else load_cifar()

print('\n--- Processing the dataset ---')                               # pre process dataset
dataset = preprocess(dataset)

print('\n--- Building the model ---')                                   # build model
model = Network()
model.build_model(dataset_name)

print('\n--- Training the model ---')                                   # train model
model.train(
    dataset,
    num_epochs,
    learning_rate,
    validate,
    regularization,
    plot_weights,
    verbose
)

print('\n--- Testing the model ---')                                    # test model
model.evaluate(
    dataset['test_images'],
    dataset['test_labels'],
    regularization,
    plot_correct,
    plot_missclassified,
    plot_feature_maps,
    verbose
)
