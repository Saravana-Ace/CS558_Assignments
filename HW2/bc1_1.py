import pickle
import numpy as np
import os

environments = ['Ant-v4', 'Walker2d-v4', 'Hopper-v4', 'HalfCheetah-v4']

for env_name in environments:
    data_path = f'cs558/expert_data/expert_data_{env_name}.pkl'
    
    with open(data_path, 'rb') as f:
        expert_data = pickle.load(f)
    

    total_rewards = [np.sum(path["reward"]) for path in expert_data]
    
    mean_return = np.mean(total_rewards)
    std_return = np.std(total_rewards)
    
    print("---------------------")
    print(f"Environment: {env_name}")
    print(f"Number of trajectories: {len(total_rewards)}")
    print(f"Mean return: {mean_return:.2f}")
    print(f"Standard deviation: {std_return:.2f}")
    print(f"Individual total_rewards: {total_rewards}")
    
print("---------------------")