import gymnasium as gym
import numpy as np
import gymnasium_robotics
from collections import defaultdict

# Create environment with dense rewards
env = gym.make('PointMaze_UMazeDense-v3', render_mode="human")

# Discretize the state and action space for simple Q-learning
def discretize_state(state, bins=(10, 10, 10, 10)):
    # Convert continuous observation to discrete state index
    pos = state['achieved_goal']
    vel = state['observation'][2:4]
    goal = state['desired_goal']
    
    # Calculate relative position to goal
    rel_pos = pos - goal
    
    # Discretize the state values
    discrete_state = (
        np.digitize(rel_pos[0], np.linspace(-2, 2, bins[0])),
        np.digitize(rel_pos[1], np.linspace(-2, 2, bins[1])),
        np.digitize(vel[0], np.linspace(-5, 5, bins[2])),
        np.digitize(vel[1], np.linspace(-5, 5, bins[3]))
    )
    return discrete_state

# Define possible actions
actions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]

# Initialize Q-table
Q = defaultdict(lambda: np.zeros(len(actions)))

# Set hyperparameters
alpha = 0.1  # Learning rate
gamma = 0.99  # Discount factor
epsilon = 0.1  # Exploration rate

# Training loop
for episode in range(1000):
    obs, _ = env.reset()
    state = discretize_state(obs)
    total_reward = 0
    done = False
    
    while not done:
        # Epsilon-greedy action selection
        if np.random.random() < epsilon:
            action_idx = np.random.choice(len(actions))
        else:
            action_idx = np.argmax(Q[state])
        
        action = actions[action_idx]
        
        # Take action
        next_obs, reward, terminated, truncated, _ = env.step(action)
        next_state = discretize_state(next_obs)
        done = terminated or truncated
        
        # Update Q-value
        Q[state][action_idx] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state][action_idx])
        
        state = next_state
        total_reward += reward
    
    print(f"Episode {episode}, Total Reward: {total_reward}")