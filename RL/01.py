# %%
import random

# %%
class Environment:
  def __init__(self):
    self.remaining_steps = 100

  # приклад спостереження
  def get_observation(self):
    return [1.0, 2.0, 1.0]  
  
  # приклад винагороди за дію
  def get_actions(self):
    return [-1, 1]

  def check_is_done(self) -> bool:
    return self.remaining_steps == 0

  def action(self,int):
    if self.check_is_done():
      raise Exception("Game over")      
    self.remaining_steps-=1
    return random.random()

# %%
class Agent:
  def __init__(self):
    self.total_rewards = 0.0 # без винагоди

  def step(self,ob: Environment):
    curr_obs = ob.get_observation()
    print(curr_obs)
    curr_action = ob.get_actions()
    print(curr_action)
    curr_reward = ob.action(random.choice(curr_action)) 
    self.total_rewards += curr_reward
    print("Total rewards so far= %.3f "%self.total_rewards)
    
# %%

obj = Environment()
agent = Agent()
step_number = 0

while not obj.check_is_done():
    step_number += 1
    agent.step(obj)

print("Total reward is %.3f "%agent.total_rewards)
# %%
