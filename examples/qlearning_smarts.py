"""This is an example to show how SMARTS multi-agent works with Q-learning. Multiple agents
learn optimal policies through reinforcement learning."""
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


class QLearningAgent(Agent):
    def __init__(self, action_space, agent_id, learning_rate=0.1, discount_factor=0.95, exploration_rate=1.0, exploration_decay=0.995):
        self._action_space = action_space
        self.agent_id = agent_id
        self.learning_rate = learning_rate  # Alpha
        self.discount_factor = discount_factor  # Gamma
        self.exploration_rate = exploration_rate  # Epsilon
        self.exploration_min = 0.01
        self.exploration_decay = exploration_decay
        self.q_table = defaultdict(lambda: np.zeros(self._action_space.n))
        self.last_state = None
        self.last_action = None
        self.episode_reward = 0
        self.training_rewards = []
        
    def _discretize_state(self, obs):
        """Convert continuous observation to discrete state for Q-table lookup."""
        # Extract relevant features from observation
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
    
    def act(self, obs, **kwargs):
        """Select action using epsilon-greedy policy."""
        current_state = self._discretize_state(obs)
        
        # Store state for learning
        self.last_state = current_state
        
        # Epsilon-greedy action selection
        if random.random() < self.exploration_rate:
            # Exploration: choose random action
            action = self._action_space.sample()
        else:
            # Exploitation: choose best action from Q-table
            action_values = self.q_table[current_state]
            action = np.argmax(action_values)
            
        # Store action for learning
        self.last_action = action
        return action
    
    def learn(self, next_obs, reward, terminated, truncated):
        """Update Q-values using the Q-learning update rule."""
        if self.last_state is None or self.last_action is None:
            return
            
        next_state = self._discretize_state(next_obs)
        done = terminated or truncated
        
        # Q-learning update formula
        # Q(s,a) = Q(s,a) + α[r + γ * max_a' Q(s',a') - Q(s,a)]
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
    plt.title('Training Rewards over Episodes')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


# def main(scenarios, headless, num_episodes, max_episode_steps=None):
#     # This interface must match the action returned by the agent
#     agent_interfaces = {
#         agent_id: AgentInterface.from_type(
#             AgentType.Laner, max_episode_steps=max_episode_steps
#         )
#         for agent_id in AGENT_IDS
#     }

#     env = gym.make(
#         "smarts.env:hiway-v1",
#         scenarios=scenarios,
#         agent_interfaces=agent_interfaces,
#         headless=headless,
#     )

#     # Create Q-learning agents
#     agents = {
#         agent_id: QLearningAgent(
#             env.action_space[agent_id],
#             agent_id,
#             learning_rate=0.1,
#             discount_factor=0.95,
#             exploration_rate=1.0
#         )
#         for agent_id in agent_interfaces.keys()
#     }
    
#     # Training loop
#     for episode in episodes(n=num_episodes):
#         observations, _ = env.reset()
#         episode.record_scenario(env.unwrapped.scenario_log)
        
#         episode_rewards = {agent_id: 0 for agent_id in agents}
#         terminateds = {"__all__": False}
#         truncateds = {"__all__": False}
        
#         # Episode loop
#         step = 0
#         while not (terminateds["__all__"] or truncateds["__all__"]):
#             # Select actions
#             actions = {
#                 agent_id: agent.act(observations[agent_id]) 
#                 for agent_id, agent in agents.items()
#             }
            
#             # Execute actions
#             next_observations, rewards, terminateds, truncateds, infos = env.step(actions)
            
#             # Learn from experience
#             for agent_id, agent in agents.items():
#                 if agent_id in next_observations:
#                     agent.learn(
#                         next_observations[agent_id],
#                         rewards[agent_id],
#                         terminateds.get(agent_id, False),
#                         truncateds.get(agent_id, False)
#                     )
#                     episode_rewards[agent_id] += rewards[agent_id]
            
#             # Update observations
#             observations = next_observations
#             step += 1
            
#             # Record step for visualization
#             episode.record_step(observations, rewards, terminateds, truncateds, infos)
        
#         # End of episode logging
#         print(f"Episode {episode.index} completed with {step} steps")
#         for agent_id, reward in episode_rewards.items():
#             print(f"  {agent_id} total reward: {reward:.2f}, exploration rate: {agents[agent_id].exploration_rate:.2f}")
    
#     # Plot learning curves
#     plot_learning_curve(agents, save_path="q_learning_training_curve.png")
#     print(f"Learning curve saved as q_learning_training_curve.png")
    
#     env.close()
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

    # Create Q-learning agents
    agents = {
        agent_id: QLearningAgent(
            env.action_space[agent_id],
            agent_id,
            learning_rate=0.1,
            discount_factor=0.95,
            exploration_rate=1.0
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
    plot_learning_curve(agents, save_path="q_learning_training_curve.png")
    print(f"Learning curve saved as q_learning_training_curve.png")
    
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