# Initialize G randomly
# Repeat for number of episodes
#  While game is not over
#   Get state and reward from env
#   Select action
#   Update env
#   Get updated state and reward
#   Store new state and reward in memory
#  Replay memory of previous episode to update G
#  G_state = G_state + α(target — G_state)  

# %%
import numpy as np
import matplotlib.pyplot as plt

# %%
ACTIONS = {'U': (-1, 0), 'D': (1, 0), 'L': (0, -1), 'R': (0, 1)}


# %%
class Maze(object):
    def __init__(self):
        self.maze = np.zeros((6, 6))
        self.maze[0, 0] = 2
        self.maze[5, :5] = 1
        self.maze[:4, 5] = 1
        self.maze[2, 2:] = 1
        self.maze[3, 2] = 1
        self.robot_position = (0, 0)
        self.steps = 0
        self.construct_allowed_states()

    def print_maze(self):
        print('--------------------------------')
        for row in self.maze:
            for col in row:
                if col == 0:
                    print('0', end="\t") # empty space
                elif col == 1:
                    print('X', end="\t") # walls
                elif col == 2:
                    print('R', end="\t") # robot position
            print("\n")
        print('---------------------------------')

    def is_allowed_move(self, state, action):
        y, x = state
        y += ACTIONS[action][0]
        x += ACTIONS[action][1]
        if y < 0 or x < 0 or y > 5 or x > 5:
            return False

        if self.maze[y, x] == 0 or self.maze[y, x] == 2:
            return True
        else:
            return False

    def construct_allowed_states(self):
        allowed_states = {}
        for y, row in enumerate(self.maze):
            for x, col in enumerate(row):
                if self.maze[(y,x)] != 1:
                    allowed_states[(y,x)] = []
                    for action in ACTIONS:
                        if self.is_allowed_move((y,x), action) & (action != 0):
                            allowed_states[(y,x)].append(action)
        self.allowed_states = allowed_states

    def update_maze(self, action):
        y, x = self.robot_position
        self.maze[y, x] = 0
        y += ACTIONS[action][0]
        x += ACTIONS[action][1]
        self.robot_position = (y, x)
        self.maze[y, x] = 2
        self.steps += 1

    def is_game_over(self):
        if self.robot_position == (5, 5):
            return True
        else:
            return False

    def get_state_and_reward(self):
        return self.robot_position, self.give_reward()

    def give_reward(self):
        if self.robot_position == (5, 5):
            return 0
        else: 
            return -1
        
# %%
class Agent(object):
    def __init__(self, states, alpha=0.15, random_factor=0.2): # 80% дослідження, 20% користування
        self.state_history = [((0, 0), 0)] # стан, винагорода
        self.alpha = alpha
        self.random_factor = random_factor
        self.G = {}
        self.init_reward(states)

    def init_reward(self, states):
        for i, row in enumerate(states):
            for j, col in enumerate(row):
                self.G[(j, i)] = np.random.uniform(low=1.0, high=0.1)
    
    def choose_action(self, state, allowed_moves):
        maxG = -10e15
        next_move = None
        randomN = np.random.random()
        if randomN < self.random_factor:
            next_move = np.random.choice(allowed_moves)
        else:
            for action in allowed_moves:
                new_state = tuple([sum(x) for x in zip(state, ACTIONS[action])])
                if self.G[new_state] >= maxG:
                    next_move = action
                    maxG = self.G[new_state]

        return next_move

    def update_state_history(self, state, reward):
        self.state_history.append((state, reward))

    def learn(self):
        target = 0

        for prev, reward in reversed(self.state_history):
            self.G[prev] = self.G[prev] + self.alpha * (target - self.G[prev])
            target += reward

        self.state_history = []

        self.random_factor -= 10e-5
        
# %%
maze = Maze()
maze.print_maze()
robot = Agent(maze.maze, alpha=0.1, random_factor=0.25)
movements_history = []

steps = 5000
for i in range(steps):
    if i % 1000 == 0:
        print(i)

    while not maze.is_game_over():
        state, _ = maze.get_state_and_reward()
        action = robot.choose_action(state, maze.allowed_states[state]) # дослідження чи користування
        maze.update_maze(action)
        state, reward = maze.get_state_and_reward()
        robot.update_state_history(state, reward)
        if maze.steps > 1000:
            maze.robot_position = (5, 5)

        if maze.steps % 100 == 0:
            print(maze.robot_position)
            print(maze.steps)

    robot.learn()
    movements_history.append(maze.steps)
    maze = Maze()
    
plt.semilogy(movements_history, "b--")
plt.show()
























# %%
# https://www.kaggle.com/code/alexisbcook/deep-reinforcement-learning
# https://www.kaggle.com/code/basu369victor/my-first-attempt-with-reinforcement-learning
# https://www.kaggle.com/learn/intro-to-game-ai-and-reinforcement-learning
# https://www.kaggle.com/code/charel/learn-by-example-reinforcement-learning-with-gym#Basic-Q-learning-algorithm
# https://www.kaggle.com/code/runway/reinforcement-learning-from-human-feedback
