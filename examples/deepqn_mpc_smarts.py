"""This is an example to show how SMARTS multi-agent works with DQN-MPC (Deep Q-Network with Model 
Predictive Control). Multiple agents learn optimal policies with enhanced lane changing behavior."""
import random
import sys
import numpy as np
from pathlib import Path
from typing import Final, Dict, List, Tuple
import gymnasium as gym
import matplotlib.pyplot as plt
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

SMARTS_REPO_PATH = Path(__file__).parents[1].absolute()
sys.path.insert(0, str(SMARTS_REPO_PATH))
from examples.tools.argument_parser import minimal_argument_parser
from smarts.core.agent import Agent
from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.utils.episodes import episodes
from smarts.sstudio.scenario_construction import build_scenarios

N_AGENTS = 4
AGENT_IDS: Final[list] = ["Agent %i" % i for i in range(N_AGENTS)]

# Neural Network for Deep Q-Learning
class DQNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# Experience Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return (
            torch.FloatTensor(state), 
            torch.LongTensor(action), 
            torch.FloatTensor(reward), 
            torch.FloatTensor(next_state),
            torch.FloatTensor(done)
        )
    
    def __len__(self):
        return len(self.buffer)

class DQNMPCAgent(Agent):
    def __init__(self, action_space, agent_id, 
                 learning_rate=0.001, discount_factor=0.95, 
                 exploration_rate=0.3, exploration_decay=0.995,
                 prediction_horizon=5, batch_size=32, 
                 target_update_freq=10):
        self._action_space = action_space
        self.agent_id = agent_id
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_min = 0.01
        self.exploration_decay = exploration_decay
        self.prediction_horizon = prediction_horizon
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # State feature dimensions
        self.state_dim = 6  # [speed, lane_index, heading_error, front_vehicle_distance, left_lane_occupied, right_lane_occupied]
        
        # DQN networks and optimizer
        self.policy_net = DQNetwork(self.state_dim, self._action_space.n)
        self.target_net = DQNetwork(self.state_dim, self._action_space.n)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer()
        
        # Internal model for predictions (simplified kinematics/dynamics model)
        self.model = self._initialize_model()
        
        self.last_state = None
        self.last_action = None
        self.episode_reward = 0
        self.training_rewards = []
        self.last_lane = None
        self.steps_done = 0
        
        # Debug counters
        self.lane_changes = 0
        self.safe_lane_changes = 0
        self.unsafe_lane_changes = 0
        self.unnecessary_lane_changes = 0
        
    def _initialize_model(self):
        """Initialize a simple prediction model for vehicle behavior."""
        return lambda state, action: self._predict_next_state(state, action)
    
    def _predict_next_state(self, state, action):
        """Predict the next state given current state and action."""
        # Extract state components
        speed, lane_index, heading_error, front_vehicle_distance, left_lane_occupied, right_lane_occupied = state
        
        # Simple state transition model
        new_speed = speed
        
        # Lane change actions
        new_lane = lane_index
        if action == 1 and lane_index > 0:  # Change left
            new_lane = lane_index - 1
        elif action == 2:  # Change right
            new_lane = lane_index + 1
            
        # Simplified heading error prediction
        new_heading_error = 0.0
        if action != 0:
            new_heading_error = 0.2 if action == 1 else -0.2
            
        # More sophisticated distance prediction
        new_distance = front_vehicle_distance
        
        # Vehicle ahead detection
        if front_vehicle_distance < 10.0:
            if action == 0:
                # If too close and not changing lanes, distance decreases
                new_distance = max(1.0, front_vehicle_distance - 1.0)
                # Speed also decreases when following too closely
                new_speed = max(0.5, speed - 0.5)
            else:
                # If changing lanes, assume distance increases (moving to a clearer lane)
                # But only if that lane is not occupied
                if (action == 1 and not left_lane_occupied) or (action == 2 and not right_lane_occupied):
                    new_distance = min(20.0, front_vehicle_distance + 3.0)
                    # Speed can be maintained or increased slightly after successful lane change
                    new_speed = min(speed + 0.3, 30.0)  # Cap at reasonable max speed
            
        return (new_speed, new_lane, new_heading_error, new_distance, left_lane_occupied, right_lane_occupied)
    
    def _predict_reward(self, state, action, next_state):
        """Predict the reward for a state-action-next_state transition."""
        # Extract state components
        speed, lane_index, _, front_vehicle_distance, left_lane_occupied, right_lane_occupied = state
        new_speed, new_lane, _, new_distance, _, _ = next_state
        
        # Base reward for making progress - proportional to speed
        reward = new_speed / 10.0  # Normalize speed reward
        
        # Incentives for lane changes when vehicle ahead
        if front_vehicle_distance < 10.0:
            # When vehicle is ahead
            if lane_index != new_lane:  # Lane change
                # Check if target lane is safe
                if (new_lane < lane_index and not left_lane_occupied) or \
                   (new_lane > lane_index and not right_lane_occupied):
                    # Reward for safe lane changes to avoid vehicle ahead
                    reward += 2.0
                else:
                    # Penalize unsafe lane change
                    reward -= 2.0
            elif action == 0 and front_vehicle_distance < 5.0:
                # Penalty for staying in lane when dangerously close
                reward -= 2.0
        else:
            # When at safe distance, penalize unnecessary lane changes
            if lane_index != new_lane:
                reward -= 0.5
            
        # Penalize being extremely close to other vehicles
        if new_distance < 1.0:
            reward -= 3.0
            
        # Penalize going out of road boundaries
        if new_lane < 0 or new_lane > 2:  # Assuming 3 lanes
            reward -= 3.0
            
        # Reward for maintaining or increasing speed
        if new_speed >= speed:
            reward += 0.3
        else:
            reward -= 0.2
            
        return reward
    
    def _discretize_state(self, obs):
        """Convert observation to feature vector for DQN."""
        if 'ego_vehicle_state' in obs:
            speed = obs['ego_vehicle_state']['speed']
            lane_index = float(obs['ego_vehicle_state'].get('lane_index', 0))
            heading_error = obs['ego_vehicle_state'].get('heading_error', 0)
            
            # Enhanced vehicle detection - get distance and relative position
            front_vehicle_distance = 20.0  # Default large value
            left_lane_occupied = 0.0
            right_lane_occupied = 0.0
            
            if 'neighborhood_vehicle_states' in obs and len(obs['neighborhood_vehicle_states']) > 0:
                # Check for vehicles ahead in same lane
                same_lane_vehicles = []
                left_lane_vehicles = []
                right_lane_vehicles = []
                
                for vehicle in obs['neighborhood_vehicle_states']:
                    rel_position = vehicle['position'][:2]
                    vehicle_lane = vehicle.get('lane_index', 0)
                    
                    # Only consider vehicles ahead (positive x direction)
                    if rel_position[0] > 0:
                        dist = np.linalg.norm(rel_position)
                        
                        if vehicle_lane == lane_index:
                            same_lane_vehicles.append((dist, vehicle))
                        elif vehicle_lane == lane_index - 1:
                            left_lane_vehicles.append((dist, vehicle))
                        elif vehicle_lane == lane_index + 1:
                            right_lane_vehicles.append((dist, vehicle))
                
                # Find closest vehicle ahead in same lane
                if same_lane_vehicles:
                    front_vehicle = min(same_lane_vehicles, key=lambda x: x[0])
                    front_vehicle_distance = front_vehicle[0]
                    
                    # Debug log
                    if front_vehicle_distance < 10.0:
                        print(f"{self.agent_id}: Detected vehicle ahead at distance {front_vehicle_distance}")
                
                # Check if adjacent lanes are occupied
                left_lane_occupied = 1.0 if any(dist < 20.0 for dist, _ in left_lane_vehicles) else 0.0
                right_lane_occupied = 1.0 if any(dist < 20.0 for dist, _ in right_lane_vehicles) else 0.0
                
                # Debug log lane occupancy
                if left_lane_occupied > 0:
                    print(f"{self.agent_id}: Left lane is occupied")
                if right_lane_occupied > 0:
                    print(f"{self.agent_id}: Right lane is occupied")
            
            # Return state vector
            return np.array([
                speed, 
                lane_index, 
                heading_error, 
                front_vehicle_distance,
                left_lane_occupied,
                right_lane_occupied
            ], dtype=np.float32)
        else:
            # Fallback for simplified observation
            return np.zeros(self.state_dim, dtype=np.float32)
    
    def _mpc_planning(self, current_state_tensor):
        """Use MPC to plan the best action sequence."""
        current_state = current_state_tensor.numpy()
        best_total_reward = float('-inf')
        best_action = 0
        
        # Emergency lane change heuristic
        lane_index = int(current_state[1])
        front_vehicle_distance = current_state[3]
        left_lane_occupied = current_state[4] > 0.5
        right_lane_occupied = current_state[5] > 0.5
        
        # Direct heuristic for immediate lane changes when vehicle ahead
        if front_vehicle_distance < 8.0:  # Emergency threshold
            print(f"{self.agent_id}: Emergency lane change consideration, vehicle at {front_vehicle_distance}")
            # Try to change lanes if safe
            if lane_index > 0 and not left_lane_occupied:
                print(f"{self.agent_id}: Emergency left lane change triggered")
                return 1  # Change left
            elif not right_lane_occupied:
                print(f"{self.agent_id}: Emergency right lane change triggered")
                return 2  # Change right
        
        # Try each possible initial action
        for action in range(self._action_space.n):
            total_reward = 0
            state = current_state
            
            # Simulate trajectory over the prediction horizon
            for step in range(self.prediction_horizon):
                # Predict next state
                next_state = self.model(state, action)
                
                # Predict reward
                reward = self._predict_reward(state, action, next_state)
                
                # For future steps after the first, use DQN for action selection
                if step == 0:
                    first_action = action
                else:
                    # Use policy network for action selection in future steps
                    with torch.no_grad():
                        state_tensor = torch.FloatTensor(state)
                        q_values = self.policy_net(state_tensor)
                        action = q_values.argmax().item()
                
                # Add discounted reward
                total_reward += (self.discount_factor ** step) * reward
                
                # Update state for next iteration
                state = next_state
                
                # Add terminal Q-value estimate
                if step == self.prediction_horizon - 1:
                    with torch.no_grad():
                        state_tensor = torch.FloatTensor(state)
                        terminal_q = self.policy_net(state_tensor).max().item()
                        total_reward += (self.discount_factor ** self.prediction_horizon) * terminal_q
            
            # If this action sequence has the best predicted outcome, choose its first action
            if total_reward > best_total_reward:
                best_total_reward = total_reward
                best_action = first_action
        
        # Debug log decision
        action_names = {0: "stay in lane", 1: "change left", 2: "change right"}
        print(f"{self.agent_id}: At distance {front_vehicle_distance}, chose to {action_names.get(best_action, 'unknown')}")
                
        return best_action
    
    def _optimize_model(self):
        """Update neural network parameters using batch of experiences."""
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # Sample batch from replay buffer
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.replay_buffer.sample(self.batch_size)
        
        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken
        state_action_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))
        
        # Compute V(s_{t+1}) for all next states
        next_state_values = torch.zeros(self.batch_size)
        with torch.no_grad():
            next_state_values = self.target_net(next_state_batch).max(1)[0] * (1 - done_batch)
        
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.discount_factor) + reward_batch
        
        # Compute loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients to avoid exploding gradients
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
    
    def act(self, obs, **kwargs):
        """Select action using epsilon-greedy policy with DQN and MPC planning."""
        # Get state representation
        state = self._discretize_state(obs)
        state_tensor = torch.FloatTensor(state)
        
        # Store state for learning
        self.last_state = state
        
        # Store lane for lane change detection
        self.last_lane = state[1]
        
        # Epsilon-greedy exploration
        sample = random.random()
        self.steps_done += 1
        current_exploration_rate = self.exploration_rate * 0.8  # Reduced for less random behavior
        
        if sample < current_exploration_rate:
            # Exploration: random action
            action = self._action_space.sample()
            print(f"{self.agent_id}: Taking random exploration action: {action}")
        else:
            # Exploitation with MPC planning
            action = self._mpc_planning(state_tensor)
            
        # Store action for learning
        self.last_action = action
        return action
    
    def learn(self, next_obs, reward, terminated, truncated):
        """Store experience in replay buffer and optimize DQN periodically."""
        if self.last_state is None or self.last_action is None:
            return
            
        next_state = self._discretize_state(next_obs)
        done = terminated or truncated
        
        # Extract relevant state information
        current_lane = int(self.last_state[1])
        front_vehicle_distance = self.last_state[3]
        left_lane_occupied = self.last_state[4] > 0.5
        right_lane_occupied = self.last_state[5] > 0.5
        
        new_lane = int(next_state[1])
        
        # Modify reward based on lane change behavior
        if current_lane != new_lane:  # Lane change occurred
            self.lane_changes += 1
            
            # Determine if lane change was justified
            if front_vehicle_distance < 10.0:  # Vehicle ahead
                if (new_lane < current_lane and not left_lane_occupied) or \
                   (new_lane > current_lane and not right_lane_occupied):
                    # Safe lane change to avoid vehicle in front
                    reward += 2.0
                    self.safe_lane_changes += 1
                    print(f"{self.agent_id}: Rewarded for safe lane change to avoid vehicle")
                else:
                    # Unsafe lane change
                    reward -= 2.0
                    self.unsafe_lane_changes += 1
                    print(f"{self.agent_id}: Penalized for unsafe lane change")
            else:
                # Unnecessary lane change
                reward -= 0.5
                self.unnecessary_lane_changes += 1
                print(f"{self.agent_id}: Penalized for unnecessary lane change")
        
        # Store transition in replay buffer
        self.replay_buffer.push(
            self.last_state,
            self.last_action,
            reward,
            next_state,
            float(done)
        )
        
        # Optimize model
        self._optimize_model()
        
        # Update target network
        if self.steps_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Track episode reward
        self.episode_reward += reward
        
        # Decay exploration rate
        if done:
            self.exploration_rate = max(
                self.exploration_min, 
                self.exploration_rate * self.exploration_decay
            )
            self.training_rewards.append(self.episode_reward)
            self.episode_reward = 0
            
    def get_training_history(self):
        """Return the training reward history."""
        return self.training_rewards
    
    def get_stats(self):
        """Get lane change statistics."""
        return {
            "lane_changes": self.lane_changes,
            "safe_lane_changes": self.safe_lane_changes,
            "unsafe_lane_changes": self.unsafe_lane_changes,
            "unnecessary_lane_changes": self.unnecessary_lane_changes
        }


def plot_learning_curve(agents, save_path=None):
    """Plot the learning curves for all agents."""
    plt.figure(figsize=(12, 8))
    
    for agent_id, agent in agents.items():
        rewards = agent.get_training_history()
        if len(rewards) > 0:  # Only plot if we have data
            # Apply moving average for smoothing
            window_size = min(10, len(rewards))
            if window_size > 0:
                smoothed_rewards = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
                plt.plot(smoothed_rewards, label=f"{agent_id}")
    
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('DQN-MPC Training Rewards')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def main(scenarios, headless, num_episodes, max_episode_steps=None):
    # This interface must match the action returned by the agent
    agent_interfaces = {
        agent_id: AgentInterface.from_type(
            AgentType.Laner, max_episode_steps=max_episode_steps
        )
        for agent_id in AGENT_IDS
    }

    env = gym.make(
        "smarts.env:hiway-v1",
        scenarios=scenarios,
        agent_interfaces=agent_interfaces,
        headless=headless,
    )

    # Create DQN-MPC agents
    agents = {
        agent_id: DQNMPCAgent(
            env.action_space[agent_id],
            agent_id,
            learning_rate=0.001,
            discount_factor=0.95,
            exploration_rate=0.3,
            prediction_horizon=5,
            batch_size=32,
            target_update_freq=10
        )
        for agent_id in agent_interfaces.keys()
    }
    
    # Training loop
    for episode in episodes(n=num_episodes):
        observations, _ = env.reset()
        episode.record_scenario(env.unwrapped.scenario_log)
        
        # Debug: Print the observation keys to understand the structure
        if episode.index == 0:
            print("Observation keys:", list(observations.keys()))
        
        episode_rewards = {agent_id: 0 for agent_id in agents}
        terminateds = {"__all__": False}
        truncateds = {"__all__": False}
        
        # Episode loop
        step = 0
        while not (terminateds["__all__"] or truncateds["__all__"]):
            # Select actions - only for agents that exist in observations
            actions = {}
            for agent_id, agent in agents.items():
                if agent_id in observations:
                    actions[agent_id] = agent.act(observations[agent_id])
                else:
                    # If agent isn't in observations, use random action as fallback
                    actions[agent_id] = env.action_space[agent_id].sample()
            
            # Execute actions
            next_observations, rewards, terminateds, truncateds, infos = env.step(actions)
            
            # Learn from experience - only for agents that exist in observations
            for agent_id, agent in agents.items():
                if agent_id in next_observations and agent_id in rewards:
                    agent.learn(
                        next_observations[agent_id],
                        rewards[agent_id],
                        terminateds.get(agent_id, False),
                        truncateds.get(agent_id, False)
                    )
                    episode_rewards[agent_id] += rewards[agent_id]
            
            # Update observations
            observations = next_observations
            step += 1
            
            # Record step for visualization
            episode.record_step(observations, rewards, terminateds, truncateds, infos)
        
        # End of episode logging
        print(f"Episode {episode.index} completed with {step} steps")
        for agent_id, reward in episode_rewards.items():
            if agent_id in agents and reward != 0:
                print(f"  {agent_id} total reward: {reward:.2f}, exploration rate: {agents[agent_id].exploration_rate:.2f}")
        
        # Print lane change statistics every 10 episodes
        if (episode.index + 1) % 10 == 0:
            print("\nLane Change Statistics:")
            for agent_id, agent in agents.items():
                stats = agent.get_stats()
                print(f"  {agent_id}: Total: {stats['lane_changes']}, Safe: {stats['safe_lane_changes']}, "
                      f"Unsafe: {stats['unsafe_lane_changes']}, Unnecessary: {stats['unnecessary_lane_changes']}")
    
    # Plot learning curves
    plot_learning_curve(agents, save_path="dqn_mpc_training_curve.png")
    print(f"Learning curve saved as dqn_mpc_training_curve.png")
    
    env.close()


if __name__ == "__main__":
    # Use the minimal_argument_parser instead of creating new arguments
    # This already has --episodes defined
    parser = minimal_argument_parser(Path(__file__).stem)
    args = parser.parse_args()

    if not args.scenarios:
        args.scenarios = [
            str(SMARTS_REPO_PATH / "scenarios" / "sumo" / "loop"),
        ]

    build_scenarios(scenarios=args.scenarios)

    main(
        scenarios=args.scenarios,
        headless=args.headless,
        num_episodes=args.episodes,
        max_episode_steps=args.max_episode_steps,
    )