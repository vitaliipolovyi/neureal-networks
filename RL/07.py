# %%
import numpy as np
import gtsam
import gtbook
import gtbook.display
from gtbook import vacuum
from gtbook.discrete import Variables

VARIABLES = Variables()
def pretty(obj):
    return gtbook.display.pretty(obj, VARIABLES)
def show(obj, **kwargs):
    return gtbook.display.show(obj, VARIABLES, **kwargs)

# %%
N = 5
X = VARIABLES.discrete_series("X", range(1, N+1), vacuum.rooms)
A = VARIABLES.discrete_series("A", range(1, N), vacuum.action_space)

conditional = gtsam.DiscreteConditional((2,5), [(0,5), (1,4)], vacuum.action_spec)
R = np.empty((5, 4, 5), float)
T = np.empty((5, 4, 5), float)
for assignment, value in conditional.enumerate():
    x, a, y = assignment[0], assignment[1], assignment[2]
    R[x, a, y] = 10.0 if y == vacuum.rooms.index("Living Room") else 0.0
    T[x, a, y] = value

# %%
def explore_randomly(x1, horizon=N):
    data = []
    x = x1
    for _ in range(1, horizon):
        a = np.random.choice(4)
        next_state_distribution = gtsam.DiscreteDistribution(X[1], T[x, a])
        x_prime = next_state_distribution.sample()
        data.append((x, a, x_prime, R[x, a, x_prime]))
        x = x_prime
    return data

# %%
data = explore_randomly(vacuum.rooms.index("Living Room"), horizon=50000)
print(data[:5])

# %%
alpha = 0.5 # learning rate
gamma = 0.9 # discount factor
Q = np.zeros((5, 4), float)
for x, a, x_prime, r in data:
    old_Q_estimate = Q[x,a]
    new_Q_estimate = r + gamma * np.max(Q[x_prime])
    Q[x, a] = (1.0-alpha) * old_Q_estimate + alpha * new_Q_estimate
print(Q)

# %%
pi_o = np.argmax(Q, axis=1)
print(f"policy = {pi_o}")
print([vacuum.action_space[a] for a in pi_o])
# %%
