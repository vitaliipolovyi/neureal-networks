# %%
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np

# %%
X, y_true = datasets.make_blobs(n_samples=150,n_features=2,
                           centers=2,cluster_std=1.05,
                           random_state=2)

# %%
plt.plot(X[:, 0][y_true == 0], X[:, 1][y_true == 0], 'r^')
plt.plot(X[:, 0][y_true == 1], X[:, 1][y_true == 1], 'bs')
plt.xlabel("feature 1")
plt.ylabel("feature 2")
plt.title('Random Classification Data with 2 classes')

# %%
def step_func(z):
        return 1.0 if (z > 0) else 0.0

def perceptron(X, y, eta, epochs):
    m, n = X.shape

    # Initializing parameters(theta) to zeros.
    # +1 in n+1 for the bias term.
    # W and b
    theta = np.zeros((n+1,1))

    # Empty list to store how many examples were
    # misclassified at every iteration.
    n_miss_list = []

    # Training.
    for epoch in range(epochs):
        # variable to store #misclassified.
        n_miss = 0

        for idx, x_i in enumerate(X):
            # Insering 1 for bias, X0 = 1.
            x_i = np.insert(x_i, 0, 1).reshape(-1, 1)

            # Calculating actvication function.
            y = step_func(np.dot(x_i.T, theta))

            # Updating if the example is misclassified.
            if abs(np.squeeze(y) - y_true[idx]) > 0.01:
                theta += eta*((y_true[idx] - y)*x_i)
                n_miss += 1

        n_miss_list.append(n_miss)

    return theta, n_miss_list

def plot_decision_boundary(X, theta):
    x1 = [min(X[:,0]), max(X[:,0])]
    m = -theta[1]/theta[2]
    c = -theta[0]/theta[2]
    x2 = m*x1 + c

    # Plotting
    fig = plt.figure(figsize=(10 ,8))
    plt.plot(X[:, 0][y_true==0], X[:, 1][y_true==0], "r^")
    plt.plot(X[:, 0][y_true==1], X[:, 1][y_true==1], "bs")
    plt.xlabel("feature 1")
    plt.ylabel("feature 2")
    plt.title("Perceptron Algorithm")
    plt.plot(x1, x2, 'y-')
    plt.show()

# %%
theta, miss_l = perceptron(X, y_true, 0.5, 5)
plot_decision_boundary(X, theta)

# %%
plt.plot(miss_l)