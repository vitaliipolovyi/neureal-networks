# %%
import numpy as np

# %%
def sigmoid(x):
  return np.tanh(x)

def dsigmoid(x):
  return 1.0-x**2

# %%
class Elman:
  def __init__(self, *args):
    self.shape = args
    n = len(args)

    # Ініціалізація шарів
    self.layers = []
    # Вхідний шарх (+1 для зміщень+розмір першого прихованого шару)
    self.layers.append(np.ones(self.shape[0]+1+self.shape[1]))
    # Приховані шари + вихідний шар
    for i in range(1,n):
      self.layers.append(np.ones(self.shape[i]))
    # Побудова матриці ваг
    self.weights = []
    for i in range(n-1):
      self.weights.append(np.zeros((self.layers[i].size,self.layers[i+1].size)))
    # dw міститиме останні зміни у вагах (для моменту)
    self.dw = [0,]*len(self.weights)

    # Скидування ваг
    self.reset()

  def reset(self):
    for i in range(len(self.weights)):
      Z = np.random.random((self.layers[i].size,self.layers[i+1].size))
      self.weights[i][...] = (2*Z-1)*0.25


  def propagate_forward(self, data):
    # Встановлення даних вхідного шару
    self.layers[0][:self.shape[0]] = data
    # Встановлення даних першого прихованого шару
    self.layers[0][self.shape[0]:-1] = self.layers[1]

    # Propagate from layer 0 to layer n-1 using sigmoid as activation function
    for i in range(1,len(self.shape)):
      # Propagate activity
      self.layers[i][...] = sigmoid(np.dot(self.layers[i-1],self.weights[i-1]))

    # Прямий хід
    return self.layers[-1]

  def propagate_backward(self, target, lrate=0.1, momentum=0.1):
    deltas = []

    # Обчислення помилки вихідного шару
    error = target - self.layers[-1]
    delta = error*dsigmoid(self.layers[-1])
    deltas.append(delta)

    # Обчислення помилки прихованих шарів
    for i in range(len(self.shape)-2,0,-1):
      delta = np.dot(deltas[0],self.weights[i].T)*dsigmoid(self.layers[i])
      deltas.insert(0,delta)
        
    # Оновлення ваг
    for i in range(len(self.weights)):
      layer = np.atleast_2d(self.layers[i])
      delta = np.atleast_2d(deltas[i])
      dw = np.dot(layer.T,delta)
      self.weights[i] += lrate*dw + momentum*self.dw[i]
      self.dw[i] = dw

    return (error**2).sum()

# %%
if __name__ == '__main__':
  # Навчання на простому часовому ряді
  network = Elman(4,8,4)
  print(network)
  samples = np.zeros(6, dtype=[('input',  float, 4), ('output', float, 4)])
  samples[0]  = (1,0,0,0), (0,1,0,0)
  samples[1]  = (0,1,0,0), (0,0,1,0)
  samples[2]  = (0,0,1,0), (0,0,0,1)
  samples[3]  = (0,0,0,1), (0,0,1,0)
  samples[4]  = (0,0,1,0), (0,1,0,0)
  samples[5]  = (0,1,0,0), (1,0,0,0)
  for i in range(5000):
    n = i % samples.size
    network.propagate_forward(samples['input'][n])
    network.propagate_backward(samples['output'][n])
  for i in range(samples.size):
    o = network.propagate_forward( samples['input'][i] )
    print('Sample %d: %s -> %s' % (i, samples['input'][i], samples['output'][i]))
    print('               Network output: %s' % (o == o.max()).astype(float))
    print('')