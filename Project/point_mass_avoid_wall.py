import gymnasium as gym
import numpy as np
import gymnasium_robotics
import time

# Create environment with dense rewards
env = gym.make('PointMaze_UMazeDense-v3', render_mode="human")
observation, info = env.reset()

# Enhanced parameters for our controller
Kp = 0.5  # Position gain
Kd = 0.8  # Velocity gain
Kr = 2.0  # Wall repulsion gain
sensor_range = 0.5  # How far to sense walls

# Stuck detection parameters
stuck_threshold = 0.08  # Movement threshold
stuck_patience = 15    # Steps to check
escape_duration = 30   # How long to apply escape behavior
exploration_angles = 8  # Different angles to try when stuck
goal_proximity_threshold = 0.6  # Don't consider "stuck" when near goal

# Tracking variables
last_positions = []
stuck_counter = 0
escape_counter = 0
wall_bias = np.array([0.0, 0.0])
current_escape_angle = 0

for step in range(1000):
    current_pos = observation['achieved_goal']
    current_vel = observation['observation'][2:4]
    goal_pos = observation['desired_goal']
    
    # Calculate distance to goal
    distance_to_goal = np.linalg.norm(goal_pos - current_pos)
    
    # Store position history
    last_positions.append(current_pos.copy())
    if len(last_positions) > stuck_patience:
        last_positions.pop(0)
    
    # Default action components
    position_error = goal_pos - current_pos
    velocity_term = -current_vel
    
    # Detect if we're stuck against a wall
    if len(last_positions) >= stuck_patience:
        total_movement = np.linalg.norm(last_positions[-1] - last_positions[0])
        
        # Only consider stuck if we're not near the goal
        stuck = (total_movement < stuck_threshold) and (distance_to_goal > goal_proximity_threshold)
        
        if stuck:
            stuck_counter += 1
            
            # If we're still stuck after trying one angle, try another
            if stuck_counter % escape_duration == 0:
                current_escape_angle = (current_escape_angle + 1) % exploration_angles
                angle = 2 * np.pi * current_escape_angle / exploration_angles
                wall_bias = np.array([np.cos(angle), np.sin(angle)])
                print(f"Still stuck! Trying new escape angle: {angle:.2f} radians")
            
            # Manage escape sequence
            if escape_counter < escape_duration:
                escape_counter += 1
            else:
                escape_counter = 0
                wall_bias += np.random.normal(0, 0.3, size=2)
                wall_bias = wall_bias / (np.linalg.norm(wall_bias) + 1e-6)
            
            # Increase strength the longer we're stuck
            escape_strength = min(1.0 + (escape_counter / 10.0), 3.0)
            
            print(f"Stuck for {stuck_counter} steps. Escape direction: {wall_bias}, Strength: {escape_strength:.2f}")
        else:
            # Not stuck, reset counters
            stuck_counter = 0
            escape_counter = 0
            wall_bias = np.array([0.0, 0.0])
    
    # Calculate action
    if escape_counter > 0:
        # When escaping, prioritize the escape direction
        escape_coefficient = escape_strength * Kr
        goal_coefficient = max(0.1, 1.0 - (escape_counter / escape_duration))
        
        action = np.clip(
            goal_coefficient * Kp * position_error + 
            Kd * velocity_term + 
            escape_coefficient * wall_bias,
            -1, 1
        )
    else:
        # Normal operation - goal seeking with velocity damping
        action = np.clip(
            Kp * position_error + 
            Kd * velocity_term,
            -1, 1
        )
    
    observation, reward, terminated, truncated, info = env.step(action)
    print(f"Position: {observation['achieved_goal']}, Goal: {observation['desired_goal']}, Reward: {reward}, Dist to goal: {distance_to_goal:.2f}")
    
    if terminated or truncated:
        observation, info = env.reset()
        last_positions = []
        stuck_counter = 0
        escape_counter = 0
        wall_bias = np.array([0.0, 0.0])
    
    time.sleep(0.01)  # For visualization

env.close()