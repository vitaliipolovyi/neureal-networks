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
def Q_value(V, x, a, gamma=0.9):
    return T[x,a] @ (R[x,a] + gamma * V)

# %%
def update_policy(value_function):
    new_policy = [None for _ in range(5)]
    for x, room in enumerate(vacuum.rooms):
        Q_values = [Q_value(value_function, x, a) for a in range(4)]
        new_policy[x] = np.argmax(Q_values)
    return new_policy

# %%
def policy_iteration(pi=None, max_iterations=100):
    for _ in range(max_iterations):
        value_for_pi = vacuum.calculate_value_function(R, T, pi) if pi is not None else np.zeros((5,))
        new_policy = update_policy(value_for_pi)
        if new_policy == pi:
            return pi, value_for_pi
        pi = new_policy
    raise RuntimeError("No stable policy found after {max_iterations} iterations")

# %%
RIGHT = vacuum.action_space.index("R")

always_right = [RIGHT, RIGHT, RIGHT, RIGHT, RIGHT]

# %%
optimal_policy, optimal_value_function = policy_iteration(always_right)
print([vacuum.action_space[a] for a in optimal_policy])

# %%
for i,room in enumerate(vacuum.rooms):
    print(f"  {room:12}: {optimal_value_function[i]:.2f}")

# %%    
optimal_policy, _ = policy_iteration()
print([vacuum.action_space[a] for a in optimal_policy])

# %%
V_k = np.full((5,), 100)
for k in range(10):
    Q_k = np.sum(T * (R + 0.9 * V_k), axis=2) # 5 x 4
    V_k = np.max(Q_k, axis=1) # max over actions
    print(np.round(V_k,2))
print(Q_k)   
# %%
print(np.round(optimal_value_function, 2))
# %%

# %%
Q_k = np.sum(T * (R + 0.9 * V_k), axis=2)
pi_k = np.argmax(Q_k, axis=1)
print(f"policy = {pi_k}")
print([vacuum.action_space[a] for a in pi_k])

