import numpy as np
import matplotlib.pyplot as plt

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def cross_entropy(x):
    return -np.log(x)


def regularized_cross_entropy(layers, lam, x):
    loss = cross_entropy(x)
    for layer in layers:
        loss += lam * (np.linalg.norm(layer.get_weights()) ** 2)
    return loss

def relu(x):
    return 0 if x < 0 else x

def leaky_relu(x, alpha=0.001):
    return x * alpha if x < 0 else x

def leaky_relu_derivative(x, alpha=0.01):
    return alpha if x < 0 else 1

def minmax_normalize(x):
    min_val = np.min(x)
    max_val = np.max(x)
    x = (x - min_val) / (max_val - min_val)
    return x

def lr_update(learning_rate, iteration): # updated code , this will fix the error 
    if iteration == 0:
        return learning_rate
    if (iteration >= 0) and (iteration <= 10000):
        return learning_rate
    if iteration > 10000:
        return learning_rate * 0.1
    if iteration > 30000:
        return learning_rate * 0.1

def preprocess(dataset):
    dataset['train_images'] = np.array([minmax_normalize(x) for x in dataset['train_images']])
    dataset['validation_images'] = np.array([minmax_normalize(x) for x in dataset['validation_images']])
    dataset['test_images'] = np.array([minmax_normalize(x) for x in dataset['test_images']])
    return dataset


def plot_accuracy_curve(accuracy_history, val_accuracy_history):
    plt.plot(accuracy_history, 'b', linewidth=3.0, label='Training accuracy')
    plt.plot(val_accuracy_history, 'r', linewidth=3.0, label='Validation accuracy')
    plt.xlabel('Iteration', fontsize=16)
    plt.ylabel('Accuracy rate', fontsize=16)
    plt.legend()
    plt.title('Training Accuracy', fontsize=16)
    plt.savefig('training_accuracy.png')
    plt.show()


def plot_learning_curve(loss_history):
    plt.plot(loss_history, 'b', linewidth=3.0, label='Cross entropy')
    plt.xlabel('Iteration', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.legend()
    plt.title('Learning Curve', fontsize=16)
    plt.savefig('learning_curve.png')
    plt.show()


def plot_sample(image, true_label, predicted_label):
    plt.imshow(image)
    if true_label and predicted_label is not None:
        if type(true_label) == 'int':
            plt.title('True label: %d, Predicted Label: %d' % (true_label, predicted_label))
        else:
            plt.title('True label: %s, Predicted Label: %s' % (true_label, predicted_label))
    plt.show()


def plot_histogram(layer_name, layer_weights):
    plt.hist(layer_weights)
    plt.title('Histogram of ' + str(layer_name))
    plt.xlabel('Value')
    plt.ylabel('Number')
    plt.show()


def to_gray(image_name):
    image = cv2.imread(image_name + '.png')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Gray image', image)
    cv2.imwrite(image_name + '.png', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def display_image(img, cmap=None):

    fig, ax = plt.subplots(nrows=1, ncols=1);

    fig.set_figwidth(8);
    fig.set_figheight(8);

    ax[0].imshow(img, cmap=cmap);
