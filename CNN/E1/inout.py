import os
import matplotlib.pyplot as plt
#import seaborn as sns
import idx2numpy
import numpy as np
#from six.moves import cPickle
import platform
#import cv2
#sns.set(color_codes=True)

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#def to_gray(image_name):
#    image = cv2.imread(image_name + '.png')
#    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#    cv2.imshow('Gray image', image)
#    cv2.imwrite(image_name + '.png', image)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()

def load_mnist():
    X_train = idx2numpy.convert_from_file('MNIST_data/train-images.idx3-ubyte')
    train_labels = idx2numpy.convert_from_file('MNIST_data/train-labels.idx1-ubyte')
    X_test = idx2numpy.convert_from_file('MNIST_data/t10k-images.idx3-ubyte')
    test_labels = idx2numpy.convert_from_file('MNIST_data/t10k-labels.idx1-ubyte')

    train_images = []                                                   # reshape train images so that the training set
    for i in range(X_train.shape[0]):                                   # is of shape (60000, 1, 28, 28)
        train_images.append(np.expand_dims(X_train[i], axis=0))
    train_images = np.array(train_images)

    test_images = []                                                    # reshape test images so that the test set
    for i in range(X_test.shape[0]):                                    # is of shape (10000, 1, 28, 28)
        test_images.append(np.expand_dims(X_test[i], axis=0))
    test_images = np.array(test_images)

    indices = np.random.permutation(train_images.shape[0])              # permute and split training data in
    training_idx, validation_idx = indices[:55000], indices[55000:]     # training and validation sets
    train_images, validation_images = train_images[training_idx, :], train_images[validation_idx, :]
    train_labels, validation_labels = train_labels[training_idx], train_labels[validation_idx]

    return {
        'train_images': train_images,
        'train_labels': train_labels,
        'validation_images': validation_images,
        'validation_labels': validation_labels,
        'test_images': test_images,
        'test_labels': test_labels
    }


#def load_pickle(f):
#    version = platform.python_version_tuple()
#    if version[0] == '2':
#        return cPickle.load(f)
#    elif version[0] == '3':
#        return cPickle.load(f, encoding='latin1')
#    raise ValueError("invalid python version: {}".format(version))


#def load_CIFAR_batch(filename):
#    X_batch = []
#    with open(filename, 'rb') as f:
#        datadict = load_pickle(f)
#        for i in range(datadict['data'].shape[0]):
#            X_batch.append(np.reshape(datadict['data'][i], (3, 32, 32)))
#        return np.array(X_batch), np.array(datadict['labels'])


#def load_cifar():
#    X_train, y_train = [], []
#    for batch in range(1, 6):
#        X_batch, y_batch = load_CIFAR_batch(os.path.join('CIFAR_data', 'data_batch_%d' % batch))
#        X_train.append(X_batch)
#        y_train.append(y_batch)
#    X_train = np.concatenate(X_train)
#    y_train = np.concatenate(y_train)
#    X_test, y_test = load_CIFAR_batch(os.path.join('CIFAR_data', 'test_batch'))
#
#    indices = np.random.permutation(X_train.shape[0])                       # permute and split training data in
#    training_idx, validation_idx = indices[:49000], indices[49000:]         # training and validation sets
#    X_train, X_val = X_train[training_idx, :], X_train[validation_idx, :]
#    y_train, y_val = y_train[training_idx], y_train[validation_idx]

#    return {
#        'train_images': X_train,
#        'train_labels': y_train,
#        'validation_images': X_val,
#        'validation_labels': y_val,
#        'test_images': X_test,
#        'test_labels': y_test
#    }
