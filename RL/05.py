# %%
import gtsam
from gtbook import vacuum
import numpy as np
from gtbook.discrete import Variables

# %%
VARIABLES = Variables()

R = np.empty((5, 4, 5), float)
N = 5
X = VARIABLES.discrete_series("X", range(1, N+1), vacuum.rooms)
T = np.empty((5, 4, 5), float)

conditional = gtsam.DiscreteConditional((2,5), [(0,5), (1,4)], vacuum.action_spec)
for assignment, value in conditional.enumerate():
    x, a, y = assignment[0], assignment[1], assignment[2]
    R[x, a, y] = 10.0 if y == vacuum.rooms.index("Living Room") else 0.0
    T[x, a, y] = value
# %%
def explore_randomly(x1, tries = N):
    data = []
    x = x1
    
    for _ in range(1, tries):
        a = np.random.choice(4)
        next_state_distribution = gtsam.DiscreteDistribution(X[1], T[x, a])
        x_prime = next_state_distribution.sample()
        data.append((x, a, x_prime, R[x, a, x_prime]))
        x = x_prime
        
    return data

data = explore_randomly(vacuum.rooms.index("Living Room"), tries=10000)
print(data[:5])
# %%
R_sum = np.zeros((5, 4, 5), float)
T_count = np.zeros((5, 4, 5), float)
count = np.zeros((5, 4), int)
for x, a, x_prime, r in data:
    R_sum[x, a, x_prime] += r
    T_count[x, a, x_prime] += 1
R_estimate = np.divide(R_sum, T_count, where=T_count!=0)
xa_count = np.sum(T_count, axis=2)
T_estimate = T_count/np.expand_dims(xa_count, axis=-1)

# %%
xa_count

# %%
print(f"real:\n{T[0]}")
print(f"estimate:\n{np.round(T_estimate[0],2)}")

# %%
print(f"real:\n{R[0]}")
print(f"estimate:\n{np.round(R_estimate[0],2)}")
# %%
