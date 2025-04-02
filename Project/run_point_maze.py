import gymnasium as gym
import numpy as np

# Register gymnasium-robotics environments
import gymnasium_robotics

# Create the environment - choose one of the available maze types
# env = gym.make('PointMaze_UMaze-v3', render_mode="human")
env = gym.make('PointMaze_UMazeDense-v3', render_mode="human")

# Reset the environment to start a new episode
observation, info = env.reset()

# Run an episode
for _ in range(2000):
    # # Take a random action
    # action = env.action_space.sample()
    
    # # Step the environment
    # observation, reward, terminated, truncated, info = env.step(action)
    
    # # Print some information
    # print(f"Position: {observation['achieved_goal']}, Goal: {observation['desired_goal']}, Reward: {reward}")
    
    # # Check if episode is done
    # if terminated or truncated:
    #     observation, info = env.reset()

    # # move toward goal heuristic will allow ball to go to goal better than random actions
    # # one issue is that the ball oscillates
    # # Simple heuristic: move toward the goal
    # current_pos = observation['achieved_goal']
    # goal_pos = observation['desired_goal']
    
    # # Calculate direction vector to goal
    # direction = goal_pos - current_pos
    # # Normalize and use as action (with some noise for exploration)
    # action = np.clip(direction + np.random.normal(0, 0.2, size=2), -1, 1)
    
    # # Step the environment
    # observation, reward, terminated, truncated, info = env.step(action)
    # print(f"Position: {observation['achieved_goal']}, Goal: {observation['desired_goal']}, Reward: {reward}")
    
    # if terminated or truncated:
    #     observation, info = env.reset()

    # solves the oscillating issue but it just takes a direct path from start to goal node without
    # taking into account of obstacles
    current_pos = observation['achieved_goal']
    current_vel = observation['observation'][2:4]  # Extract velocity from observation
    goal_pos = observation['desired_goal']
    
    # Proportional term (position error)
    position_error = goal_pos - current_pos
    
    # Derivative term (velocity)
    velocity_term = -current_vel  # We want to counteract current velocity
    
    # Calculate action using both terms
    action = np.clip(0.5 * position_error + 0.8 * velocity_term, -1, 1)
    
    observation, reward, terminated, truncated, info = env.step(action)
    print(f"Position: {observation['achieved_goal']}, Goal: {observation['desired_goal']}, Reward: {reward}")

    if terminated or truncated:
        observation, info = env.reset()
    

# Close the environment
env.close()