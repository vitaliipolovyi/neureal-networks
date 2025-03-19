# %%
from inout import load_mnist
from utils import preprocess, regularized_cross_entropy
from network import Network
import numpy as np

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

# %%
print('\n--- Loading ' + dataset_name + ' dataset ---')                 # load dataset
dataset = load_mnist() # if dataset_name is 'mnist' else load_cifar()

# %%
print('\n--- Processing the dataset ---')                               # pre process dataset
dataset = preprocess(dataset)

# %%
print('\n--- Building the model ---')                                   # build model
model = Network()
model.build_model(dataset_name)

# %%
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

# %%
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
# %%

# %%
from utils import regularized_cross_entropy, plot_learning_curve, plot_accuracy_curve, plot_histogram, plot_sample, lr_update
import numpy as np

def forward(image, plot_feature_maps):                # forward propagate
    for layer in model.layers:
        if plot_feature_maps:
            plot_sample((image * 255)[0, :, :], None, None)
        image = layer.forward(image)

    return image
  
def evaluate(X, y, regularization, plot_correct, plot_missclassified, plot_feature_maps, verbose):
    loss, num_correct = 0, 0
    for i in range(len(X)):
        tmp_output = forward(X[i], plot_feature_maps)              # forward propagation

        # compute cross-entropy update loss
        loss += regularized_cross_entropy(model.layers, regularization, tmp_output[y[i]])

        prediction = np.argmax(tmp_output)                              # update accuracy
        if prediction == y[i]:
            num_correct += 1
            if plot_correct:                                            # plot correctly classified digit
                image = (X[i] * 255)[0, :, :]
                plot_sample(image, y[i], prediction)
                plot_correct = 1
        else:
            if plot_missclassified:                                     # plot missclassified digit
                image = (X[i] * 255)[0, :, :]
                plot_sample(image, y[i], prediction)
                plot_missclassified = 1

    test_size = len(X)
    accuracy = (num_correct / test_size) * 100
    loss = loss / test_size
    if verbose:
        print('Test Loss: %02.3f' % loss)
        print('Test Accuracy: %02.3f' % accuracy)
    return loss, accuracy

evaluate(
    dataset['test_images'],
    dataset['test_labels'],
    regularization,
    plot_correct,
    plot_missclassified,
    plot_feature_maps,
    verbose   
)
# %%
