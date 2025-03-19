# %%
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# %%
# Define the fuzzy variables and membership functions
temperature = ctrl.Antecedent(np.arange(0, 101, 1), 'temperature')
humidity = ctrl.Antecedent(np.arange(0, 101, 1), 'humidity')
fan_speed = ctrl.Consequent(np.arange(0, 101, 1), 'fan_speed')

# Define the membership functions for each variable
temperature['low'] = fuzz.trimf(temperature.universe, [0, 0, 50])
temperature['medium'] = fuzz.trimf(temperature.universe, [0, 50, 100])
temperature['high'] = fuzz.trimf(temperature.universe, [50, 100, 100])

humidity['low'] = fuzz.trimf(humidity.universe, [0, 0, 50])
humidity['medium'] = fuzz.trimf(humidity.universe, [0, 50, 100])
humidity['high'] = fuzz.trimf(humidity.universe, [50, 100, 100])

fan_speed['low'] = fuzz.trimf(fan_speed.universe, [0, 0, 50])
fan_speed['medium'] = fuzz.trimf(fan_speed.universe, [0, 50, 100])
fan_speed['high'] = fuzz.trimf(fan_speed.universe, [50, 100, 100])

# %%
# Define the fuzzy rules
rule1 = ctrl.Rule(temperature['low'] | humidity['low'], fan_speed['low'])
rule2 = ctrl.Rule(temperature['medium'] | humidity['medium'], fan_speed['medium'])
rule3 = ctrl.Rule(temperature['high'] | humidity['high'], fan_speed['high'])

# %%
# Create the control system
fan_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])

# %%
# Create the control system simulation
fan_speed_ctrl = ctrl.ControlSystemSimulation(fan_ctrl)

# %%
# Pass inputs to the control system and compute the output
fan_speed_ctrl.input['temperature'] = 75
fan_speed_ctrl.input['humidity'] = 30
fan_speed_ctrl.compute()

# %%
# Print the computed output
print(fan_speed_ctrl.output['fan_speed'])

# %%
# Plot the membership functions and the output
temperature.view()
humidity.view()
fan_speed.view(sim=fan_speed_ctrl)