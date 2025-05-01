"""This is an example to show how SMARTS multi-agent works with Q-MPC (Q-learning with Model 
Predictive Control). Multiple agents learn optimal policies while planning ahead to reduce 
unnecessary lane changes."""
import random
import sys
import numpy as np
from pathlib import Path
from typing import Final
import gymnasium as gym
import matplotlib.pyplot as plt
from collections import defaultdict

SMARTS_REPO_PATH = Path(__file__).parents[1].absolute()
sys.path.insert(0, str(SMARTS_REPO_PATH))
from examples.tools.argument_parser import minimal_argument_parser
from smarts.core.agent import Agent
from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.utils.episodes import episodes
from smarts.sstudio.scenario_construction import build_scenarios

N_AGENTS = 4
AGENT_IDS: Final[list] = ["Agent %i" % i for i in range(N_AGENTS)]


class QMPCAgent(Agent):
    def __init__(self, action_space, agent_id, learning_rate=0.1, discount_factor=0.95, 
                 exploration_rate=0.3, exploration_decay=0.995, prediction_horizon=5):
        self._action_space = action_space
        self.agent_id = agent_id
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_min = 0.01
        self.exploration_decay = exploration_decay
        self.prediction_horizon = prediction_horizon  # Planning horizon for MPC
        
        # Q-table for value estimation
        self.q_table = defaultdict(lambda: np.zeros(self._action_space.n))
        
        # Internal model for predictions (simplified kinematics/dynamics model)
        self.model = self._initialize_model()
        
        self.last_state = None
        self.last_action = None
        self.episode_reward = 0
        self.training_rewards = []
        self.last_lane = None  # Track the last lane to detect lane changes
        
    def _initialize_model(self):
        """Initialize a simple prediction model for vehicle behavior."""
        # This could be a learned model, but we'll use a simplified kinematic model
        return lambda state, action: self._predict_next_state(state, action)
    
    def _predict_next_state(self, state, action):
        """Predict the next state given current state and action."""
        # Extract state components
        speed, lane_index, heading_error, distance_to_vehicle = state
        
        # Simple state transition model (this would be more complex in a real implementation)
        new_speed = speed  # Assume speed remains relatively constant
        
        # Lane change actions (assuming discrete actions)
        # Action mapping: 0=maintain lane, 1=change left, 2=change right, etc.
        new_lane = lane_index
        if action == 1 and lane_index > 0:  # Change left
            new_lane = lane_index - 1
        elif action == 2:  # Change right
            new_lane = lane_index + 1
            
        # Simple prediction for heading error (would be more complex in reality)
        new_heading_error = 0.0
        if action != 0:  # If changing lanes, temporarily increase heading error
            new_heading_error = 0.2 if action == 1 else -0.2
            
        # Simplistic prediction for distance to vehicle
        new_distance = distance_to_vehicle
        if distance_to_vehicle < 5 and action == 0:
            # If we're close to a vehicle and not changing lanes, distance might decrease
            new_distance = max(1.0, distance_to_vehicle - 1.0)
            
        return (new_speed, new_lane, new_heading_error, new_distance)
    
    def _predict_reward(self, state, action, next_state):
        """Predict the reward for a state-action-next_state transition."""
        # Extract state components
        current_lane = state[1]
        new_lane = next_state[1]
        distance_to_vehicle = next_state[3]
        
        # Base reward for making progress
        reward = 1.0
        
        # Penalize lane changes (to reduce unnecessary lane changes)
        if current_lane != new_lane:
            reward -= 0.5
            
        # Penalize being too close to other vehicles
        if distance_to_vehicle < 2.0:
            reward -= 1.0
            
        # Penalize going out of road boundaries (simplified)
        if new_lane < 0 or new_lane > 2:  # Assuming 3 lanes
            reward -= 2.0
            
        return reward
    
    def _discretize_state(self, obs):
        """Convert continuous observation to discrete state for Q-table lookup."""
        if 'ego_vehicle_state' in obs:
            speed = round(obs['ego_vehicle_state']['speed'], 1)
            lane_index = obs['ego_vehicle_state'].get('lane_index', 0)
            heading_error = round(obs['ego_vehicle_state'].get('heading_error', 0), 1)
            
            # Get distance to nearest vehicle if available
            distance_to_vehicle = 10.0  # Default large value
            if 'neighborhood_vehicle_states' in obs and len(obs['neighborhood_vehicle_states']) > 0:
                distances = [np.linalg.norm(vehicle['position'][:2]) for vehicle in obs['neighborhood_vehicle_states']]
                if distances:
                    distance_to_vehicle = round(min(distances), 1)
            
            return (speed, lane_index, heading_error, distance_to_vehicle)
        else:
            # Simplified fallback for observation formats without ego_vehicle_state
            return tuple(map(lambda x: round(x, 1), obs.values()))
    
    def _mpc_planning(self, current_state):
        """Use MPC to plan the best action sequence."""
        best_total_reward = float('-inf')
        best_action = 0
        
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
                
                # For future steps after the first, use the action with highest Q-value
                if step == 0:
                    first_action = action
                else:
                    # Use Q-values for action selection in future steps
                    action = np.argmax(self.q_table[state])
                
                # Add discounted reward
                total_reward += (self.discount_factor ** step) * reward
                
                # Update state for next iteration
                state = next_state
                
                # Add terminal Q-value estimate
                if step == self.prediction_horizon - 1:
                    total_reward += (self.discount_factor ** self.prediction_horizon) * np.max(self.q_table[state])
            
            # If this action sequence has the best predicted outcome, choose its first action
            if total_reward > best_total_reward:
                best_total_reward = total_reward
                best_action = first_action
                
        return best_action
    
    def act(self, obs, **kwargs):
        """Select action using Q-MPC approach."""
        current_state = self._discretize_state(obs)
        
        # Store state for learning
        self.last_state = current_state
        
        # Store lane for lane change detection
        self.last_lane = current_state[1]
        
        # Epsilon-greedy exploration
        if random.random() < self.exploration_rate:
            # Pure exploration: random action
            action = self._action_space.sample()
        else:
            # Exploitation with MPC planning
            action = self._mpc_planning(current_state)
            
        # Store action for learning
        self.last_action = action
        return action
    
    def learn(self, next_obs, reward, terminated, truncated):
        """Update Q-values using the Q-learning update rule."""
        if self.last_state is None or self.last_action is None:
            return
            
        next_state = self._discretize_state(next_obs)
        done = terminated or truncated
        
        # Apply custom reward modifications
        current_lane = next_state[1]
        
        # Penalize unnecessary lane changes
        if self.last_lane is not None and self.last_lane != current_lane:
            # Check if lane change was necessary (e.g., avoiding obstacle)
            was_necessary = False
            if 'neighborhood_vehicle_states' in next_obs:
                for vehicle in next_obs['neighborhood_vehicle_states']:
                    # If vehicle was too close in the previous lane, lane change was necessary
                    if np.linalg.norm(vehicle['position'][:2]) < 3.0 and vehicle.get('lane_index', -1) == self.last_lane:
                        was_necessary = True
                        break
                        
            if not was_necessary:
                # Add penalty for unnecessary lane change
                reward -= 0.5
                
        # Q-learning update formula
        current_q = self.q_table[self.last_state][self.last_action]
        
        if done:
            # If episode is done, there's no next state
            next_max_q = 0
        else:
            # Maximum Q-value for next state
            next_max_q = np.max(self.q_table[next_state])
            
        # Calculate new Q-value
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * next_max_q - current_q
        )
        
        # Update Q-table
        self.q_table[self.last_state][self.last_action] = new_q
        
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
    plt.title('Q-MPC Training Rewards over Episodes')
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

    # Create Q-MPC agents
    agents = {
        agent_id: QMPCAgent(
            env.action_space[agent_id],
            agent_id,
            learning_rate=0.1,
            discount_factor=0.95,
            exploration_rate=0.3,
            prediction_horizon=5  # Look 5 steps ahead
        )
        for agent_id in agent_interfaces.keys()
    }
    
    # Training loop
    for episode in episodes(n=num_episodes):
        observations, _ = env.reset()
        episode.record_scenario(env.unwrapped.scenario_log)
        
        # Debug: Print the observation keys to understand the structure
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
            if agent_id in agents:
                print(f"  {agent_id} total reward: {reward:.2f}, exploration rate: {agents[agent_id].exploration_rate:.2f}")
    
    # Plot learning curves
    plot_learning_curve(agents, save_path="qmpc_training_curve.png")
    print(f"Learning curve saved as qmpc_training_curve.png")
    
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