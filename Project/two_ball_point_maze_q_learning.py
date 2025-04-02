import gymnasium as gym
import numpy as np
import gymnasium_robotics
from collections import defaultdict
import time

class TwoAgentPointMaze:
    def __init__(self, render_mode="human", maze_type="PointMaze_UMazeDense-v3"):
        # Create two separate environments
        self.env1 = gym.make(maze_type, render_mode=render_mode)
        self.env2 = gym.make(maze_type, render_mode=None)  # Only render one environment
        
        # Define collision radius (distance at which balls collide)
        self.collision_radius = 0.3
        
        # Define collision penalty
        self.collision_penalty = -1.0
        
    def reset(self):
        # Reset both environments
        obs1, info1 = self.env1.reset()
        obs2, info2 = self.env2.reset()
        
        # Ensure agents start at different positions
        attempts = 0
        while self._check_collision(obs1['achieved_goal'], obs2['achieved_goal']) and attempts < 10:
            obs2, info2 = self.env2.reset()
            attempts += 1
            
        return (obs1, obs2), (info1, info2)
    
    def step(self, actions):
        action1, action2 = actions
        
        # Instead of directly accessing internal data, we'll use the observation values
        
        # Take actions in both environments
        obs1, reward1, term1, trunc1, info1 = self.env1.step(action1)
        obs2, reward2, term2, trunc2, info2 = self.env2.step(action2)
        
        # Check for collision using the achieved_goal from observations
        new_pos1 = obs1['achieved_goal']
        new_pos2 = obs2['achieved_goal']
        
        collision = self._check_collision(new_pos1, new_pos2)
        
        if collision:
            # Apply collision penalty
            reward1 += self.collision_penalty
            reward2 += self.collision_penalty
            
            # Note: We can't easily revert positions without directly accessing the simulator
            # If needed, we could implement a soft collision response instead
        
        # Combine rewards and termination conditions
        terminated = term1 or term2
        truncated = trunc1 or trunc2
        
        return (obs1, obs2), (reward1, reward2), terminated, truncated, (info1, info2)
    
    def _check_collision(self, pos1, pos2):
        # Calculate distance between balls
        distance = np.linalg.norm(pos1 - pos2)
        return distance < self.collision_radius
    
    def render(self):
        # Only render the first environment
        return self.env1.render()
    
    def close(self):
        self.env1.close()
        self.env2.close()

# Create the two-agent environment
two_ball_env = TwoAgentPointMaze(render_mode="human")

# Setup Q-learning (similar to before but adapted for two agents)
def discretize_state(state, bins=(10, 10, 10, 10)):
    pos = state['achieved_goal']
    vel = state['observation'][2:4]
    goal = state['desired_goal']
    
    # Calculate relative position to goal
    rel_pos = pos - goal
    
    discrete_state = (
        np.digitize(rel_pos[0], np.linspace(-2, 2, bins[0])),
        np.digitize(rel_pos[1], np.linspace(-2, 2, bins[1])),
        np.digitize(vel[0], np.linspace(-5, 5, bins[2])),
        np.digitize(vel[1], np.linspace(-5, 5, bins[3]))
    )
    return discrete_state

# Define possible actions
actions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]

# Initialize Q-tables for both agents
Q1 = defaultdict(lambda: np.zeros(len(actions)))
Q2 = defaultdict(lambda: np.zeros(len(actions)))

# Set hyperparameters
alpha = 0.1
gamma = 0.99
epsilon = 0.1

# Training loop
for episode in range(500):
    (obs1, obs2), _ = two_ball_env.reset()
    state1 = discretize_state(obs1)
    state2 = discretize_state(obs2)
    total_reward1 = 0
    total_reward2 = 0
    done = False
    
    while not done:
        # Select actions for both agents
        if np.random.random() < epsilon:
            action_idx1 = np.random.choice(len(actions))
        else:
            action_idx1 = np.argmax(Q1[state1])
            
        if np.random.random() < epsilon:
            action_idx2 = np.random.choice(len(actions))
        else:
            action_idx2 = np.argmax(Q2[state2])
        
        action1 = actions[action_idx1]
        action2 = actions[action_idx2]
        
        # Take actions
        (next_obs1, next_obs2), (reward1, reward2), terminated, truncated, _ = two_ball_env.step((action1, action2))
        next_state1 = discretize_state(next_obs1)
        next_state2 = discretize_state(next_obs2)
        done = terminated or truncated
        
        # Update Q-values
        Q1[state1][action_idx1] += alpha * (reward1 + gamma * np.max(Q1[next_state1]) - Q1[state1][action_idx1])
        Q2[state2][action_idx2] += alpha * (reward2 + gamma * np.max(Q2[next_state2]) - Q2[state2][action_idx2])
        
        state1 = next_state1
        state2 = next_state2
        total_reward1 += reward1
        total_reward2 += reward2
        
        # Delay for visualization
        time.sleep(0.01)
    
    print(f"Episode {episode}, Rewards: Agent1={total_reward1:.2f}, Agent2={total_reward2:.2f}")

two_ball_env.close()