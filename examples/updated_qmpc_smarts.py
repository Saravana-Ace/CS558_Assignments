"""This is an example to show how SMARTS multi-agent works with Q-MPC (Q-learning with Model 
Predictive Control). Multiple agents learn optimal policies with enhanced lane changing behavior."""
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
        
        # Debug counters
        self.lane_changes = 0
        self.safe_lane_changes = 0
        self.unsafe_lane_changes = 0
        self.unnecessary_lane_changes = 0
        
    def _initialize_model(self):
        """Initialize a simple prediction model for vehicle behavior."""
        # This could be a learned model, but we'll use a simplified kinematic model
        return lambda state, action: self._predict_next_state(state, action)
    
    def _predict_next_state(self, state, action):
        """Predict the next state given current state and action."""
        # Extract state components - handle both standard and enhanced state
        if len(state) >= 6:  # Enhanced state with adjacent lane info
            speed, lane_index, heading_error, front_vehicle_distance, left_lane_occupied, right_lane_occupied = state
        else:  # Standard state
            speed, lane_index, heading_error, front_vehicle_distance = state
            left_lane_occupied, right_lane_occupied = 0, 0
        
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
        
        # A closer threshold for "too close" - 10.0 instead of 5.0
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
        # Extract state components - handle both state formats
        if len(state) >= 6:  # Enhanced state
            speed, lane_index, _, front_vehicle_distance, left_lane_occupied, right_lane_occupied = state
            new_speed, new_lane, _, new_distance, _, _ = next_state
        else:  # Basic state
            speed, lane_index, _, front_vehicle_distance = state
            new_speed, new_lane, _, new_distance = next_state
            left_lane_occupied, right_lane_occupied = 0, 0
        
        # Base reward for making progress - proportional to speed
        reward = new_speed / 10.0  # Normalize speed reward
        
        # MODIFIED: Stronger incentives for lane changes when vehicle ahead
        # Use 10.0 as the threshold for "too close" instead of 3.0
        if front_vehicle_distance < 10.0:
            # When vehicle is ahead
            if lane_index != new_lane:  # Lane change
                # Check if target lane is safe
                if (new_lane < lane_index and not left_lane_occupied) or \
                   (new_lane > lane_index and not right_lane_occupied):
                    # STRONGER reward for safe lane changes to avoid vehicle ahead
                    reward += 2.0  # Increased from 1.0
                else:
                    # Penalize unsafe lane change
                    reward -= 2.0
            elif action == 0 and front_vehicle_distance < 5.0:
                # STRONGER penalty for staying in lane when dangerously close
                reward -= 2.0  # Increased from 1.0
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
            reward += 0.3  # Increased from 0.2
        else:
            reward -= 0.2  # Increased from 0.1
            
        return reward
    
    def _discretize_state(self, obs):
        """Convert continuous observation to discrete state for Q-table lookup with improved features."""
        if 'ego_vehicle_state' in obs:
            speed = round(obs['ego_vehicle_state']['speed'], 1)
            lane_index = obs['ego_vehicle_state'].get('lane_index', 0)
            heading_error = round(obs['ego_vehicle_state'].get('heading_error', 0), 1)
            
            # Enhanced vehicle detection - get distance and relative position
            front_vehicle_distance = 20.0  # Default large value
            left_lane_occupied = False
            right_lane_occupied = False
            
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
                        # MODIFIED: Calculate Euclidean distance for better detection
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
                    front_vehicle_distance = round(front_vehicle[0], 1)
                    
                    # PRINT DEBUG: Log when vehicle is detected ahead
                    if front_vehicle_distance < 10.0:
                        print(f"{self.agent_id}: Detected vehicle ahead at distance {front_vehicle_distance}")
                
                # Check if adjacent lanes are occupied (within overtaking distance)
                # MODIFIED: Use a larger detection range (20.0 instead of 15.0)
                left_lane_occupied = any(dist < 20.0 for dist, _ in left_lane_vehicles)
                right_lane_occupied = any(dist < 20.0 for dist, _ in right_lane_vehicles)
                
                # PRINT DEBUG: Log lane occupancy
                if left_lane_occupied:
                    print(f"{self.agent_id}: Left lane is occupied")
                if right_lane_occupied:
                    print(f"{self.agent_id}: Right lane is occupied")
            
            # Return enhanced state representation
            return (speed, lane_index, heading_error, front_vehicle_distance, 
                   int(left_lane_occupied), int(right_lane_occupied))
        else:
            # Simplified fallback
            return tuple(map(lambda x: round(x, 1), obs.values()))
    
    def _mpc_planning(self, current_state):
        """Use MPC to plan the best action sequence."""
        best_total_reward = float('-inf')
        best_action = 0
        
        # DEBUG: Extract key state elements
        if len(current_state) >= 6:
            _, lane_index, _, front_vehicle_distance, left_lane_occupied, right_lane_occupied = current_state
            
            # MODIFIED: Add direct heuristic for immediate lane changes when vehicle ahead
            # This will bypass MPC in emergency situations
            if front_vehicle_distance < 8.0:  # Emergency threshold
                print(f"{self.agent_id}: Emergency lane change consideration, vehicle at {front_vehicle_distance}")
                # Try to change lanes if safe
                if lane_index > 0 and not left_lane_occupied:
                    print(f"{self.agent_id}: Emergency left lane change triggered")
                    return 1  # Change left
                elif not right_lane_occupied:
                    print(f"{self.agent_id}: Emergency right lane change triggered")
                    return 2  # Change right
                # If no safe lane change is possible, continue with MPC
        
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
        
        # DEBUG: Log decision
        if len(current_state) >= 6:
            _, _, _, front_dist, _, _ = current_state
            action_names = {0: "stay in lane", 1: "change left", 2: "change right"}
            print(f"{self.agent_id}: At distance {front_dist}, chose to {action_names.get(best_action, 'unknown')}")
                
        return best_action
    
    def act(self, obs, **kwargs):
        """Select action using Q-MPC approach."""
        current_state = self._discretize_state(obs)
        
        # Store state for learning
        self.last_state = current_state
        
        # Store lane for lane change detection
        self.last_lane = current_state[1]
        
        # MODIFIED: Lower exploration during early episodes
        if random.random() < self.exploration_rate * 0.8:  # Reduced by 20%
            # Pure exploration: random action
            action = self._action_space.sample()
            print(f"{self.agent_id}: Taking random exploration action: {action}")
        else:
            # Exploitation with MPC planning
            action = self._mpc_planning(current_state)
            
        # Store action for learning
        self.last_action = action
        return action
    
    def learn(self, next_obs, reward, terminated, truncated):
        """Update Q-values with enhanced reward for safe lane changes."""
        if self.last_state is None or self.last_action is None:
            return
            
        next_state = self._discretize_state(next_obs)
        done = terminated or truncated
        
        # Extract relevant state information
        if len(self.last_state) >= 6:  # Enhanced state representation
            current_lane = self.last_state[1]
            front_vehicle_distance = self.last_state[3]
            left_lane_occupied = self.last_state[4]
            right_lane_occupied = self.last_state[5]
            
            new_lane = next_state[1]
            
            # MODIFIED: Stronger reward modifications based on lane change behavior
            if current_lane != new_lane:  # Lane change occurred
                self.lane_changes += 1
                
                # Determine if lane change was justified
                if front_vehicle_distance < 10.0:  # INCREASED threshold
                    if (new_lane < current_lane and not left_lane_occupied) or \
                       (new_lane > current_lane and not right_lane_occupied):
                        # Safe lane change to avoid vehicle in front
                        reward += 2.0  # INCREASED from 1.0
                        self.safe_lane_changes += 1
                        print(f"{self.agent_id}: Strongly rewarded for safe lane change to avoid vehicle")
                    else:
                        # Unsafe lane change
                        reward -= 2.0  # INCREASED from 1.0
                        self.unsafe_lane_changes += 1
                        print(f"{self.agent_id}: Strongly penalized for unsafe lane change")
                else:
                    # Unnecessary lane change
                    reward -= 0.5
                    self.unnecessary_lane_changes += 1
                    print(f"{self.agent_id}: Penalized for unnecessary lane change")
        
        # Standard Q-learning update
        current_q = self.q_table[self.last_state][self.last_action]
        
        if done:
            next_max_q = 0
        else:
            next_max_q = np.max(self.q_table[next_state])
            
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
    plt.title('Smart Lane-Changing Q-MPC Training Rewards')
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
    plot_learning_curve(agents, save_path="smart_lane_qmpc_training_curve.png")
    print(f"Learning curve saved as smart_lane_qmpc_training_curve.png")
    
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