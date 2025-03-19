# %%
import matplotlib.pyplot as plt
import numpy as np

# %%
np.random.seed(0xdeadbeef)

# %%
def make_sample(nexamples, means=([0.,0.],[1.,1.]), sigma=1.):
    normal = np.random.multivariate_normal
    # squared width:
    s2 = sigma**2.
    # below, we provide the coordinates of the mean as 
    # a first argument, and then the covariance matrix
    # which describes the width of the Gaussian along the 
    # two directions. 
    # we generate nexamples examples for each category
    sgx0 = normal(means[0], [[s2, 0.], [0.,s2]], nexamples)
    sgx1 = normal(means[1], [[s2, 0.], [0.,s2]], nexamples)
    # setting the labels for each category
    sgy0 = np.zeros((nexamples,))
    sgy1 = np.ones((nexamples,))
    sgx = np.concatenate([sgx0,sgx1])
    sgy = np.concatenate([sgy0,sgy1])
    return sgx, sgy
  
# %%
sgx, sgy = make_sample(30)
tgx, tgy = make_sample(200)

# %%
plt.scatter(sgx[:,0], sgx[:,1], alpha=0.5, c=sgy)
plt.xlabel('x1')
plt.ylabel('x2')

# %%
from sklearn.neural_network import MLPClassifier

# Starting model
# %% 
mlp = MLPClassifier(hidden_layer_sizes=(50,50,50), activation='relu', max_iter=10000, random_state=1)
mlp.fit(sgx,sgy)

# %%
def plot_result(sample, targets, linrange=(-5,5,101)):
  xmin, xmax, npoints = linrange
  gridx1, gridx2 = np.meshgrid(np.linspace(xmin,xmax,npoints), np.linspace(xmin,xmax,npoints))
  grid = np.c_[gridx1.flatten(), gridx2.flatten()]
  probs = mlp.predict_proba(grid)
  plt.pcolor(gridx1, gridx2, probs[:,1].reshape(npoints,npoints), cmap='binary')
  plt.colorbar()
  plt.scatter(sample[:,0], sample[:,1], c=targets, cmap='plasma', alpha=0.5, marker='.')
  plt.xlabel('x1')
  plt.ylabel('x2')
  plt.show()

# Training sample + normal fit
# %%
plot_result(sgx,sgy)

# Test sample + overfitting
# %%
plot_result(tgx,tgy)

# Fixing overfitting (1) - simpler model
# %%
mlp = MLPClassifier(hidden_layer_sizes=(5,), activation='relu', max_iter=10000, random_state=1)
mlp.fit(sgx,sgy)
plot_result(tgx,tgy)

# Fixing overfitting (2) - longer execution time
# %%
sgx, sgy = make_sample(10000)
mlp = MLPClassifier(hidden_layer_sizes=(50,50,50), activation='relu', max_iter=10000, random_state=1)
mlp.fit(sgx,sgy)
plot_result(tgx, tgy)

# Complex data
# %%
sgxa, sgya = make_sample(1000, ([0.,0],[3.,3.]), 0.3)
sgxb, sgyb = make_sample(1000, ([1.,1],[4.,4.]), 0.3)
sgxc, sgyc = make_sample(1000, ([5.,5.],[-2.,-2.]), 0.6)
sgxd, sgyd = make_sample(1000, ([-1,3.],[3.,-1.]), 0.3)

sgx = np.concatenate([sgxa,sgxb,sgxc,sgxd])
sgy = np.concatenate([sgya,sgyb,sgyc,sgyd])

# %%
plt.scatter(sgx[:,0], sgx[:,1], alpha=0.5, c=sgy)
plt.xlabel('x1')
plt.ylabel('x2')

# %%
mlp = MLPClassifier(hidden_layer_sizes=(3,), activation='relu', max_iter=10000, random_state=1)
mlp.fit(sgx,sgy)

# Underfitting
# %%
plot_result(sgx,sgy,linrange=(-4,7,201))

# + Neurons
# %%
mlp = MLPClassifier(hidden_layer_sizes=(5,), activation='relu', max_iter=10000, random_state=1)
mlp.fit(sgx,sgy)
plot_result(sgx,sgy,linrange=(-4,7,201))

# + Neurons
# %%
mlp = MLPClassifier(hidden_layer_sizes=(50,50,50), activation='relu', max_iter=10000, random_state=1)
mlp.fit(sgx,sgy)
plot_result(sgx,sgy,linrange=(-4,7,201))

# %%
