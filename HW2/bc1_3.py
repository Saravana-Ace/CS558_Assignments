import subprocess
import numpy as np
import matplotlib.pyplot as plt
import re
import os

env_name = "Hopper-v4"

step_values = [1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]

mean_returns = []
std_returns = []

print(f"Running hyperparameter sensitivity analysis on {env_name}")
print(f"Varying num_agent_train_steps_per_iter: {step_values}")
print("=" * 60)

for steps in step_values:
    print(f"\nTesting with {steps} training steps per iteration...")
    
    cmd = [
        'python', 'cs558/scripts/run_hw2.py',
        '--expert_policy_file', f'cs558/policies/experts/{env_name.split("-")[0]}.pkl',
        '--env_name', env_name,
        '--exp_name', f'bc_{env_name.lower().split("-")[0]}_steps_{steps}',
        '--n_iter', '1',
        '--expert_data', f'cs558/expert_data/expert_data_{env_name}.pkl',
        '--video_log_freq', '-1',
        '--train_batch_size', '1000',
        '--num_agent_train_steps_per_iter', str(steps),
        '--eval_batch_size', '5000'
    ]
    
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    output, error = process.communicate()
    
    mean_match = re.search(r'Eval_AverageReturn : ([\d\.\-]+)', output)
    std_match = re.search(r'Eval_StdReturn : ([\d\.\-]+)', output)
    
    if mean_match and std_match:
        mean = float(mean_match.group(1))
        std = float(std_match.group(1))
        mean_returns.append(mean)
        std_returns.append(std)
        print(f"Mean return: {mean:.2f}, Std: {std:.2f}")
    else:
        print("Failed to extract results")

mean_returns = np.array(mean_returns)
std_returns = np.array(std_returns)

plt.figure(figsize=(10, 6))
plt.errorbar(step_values, mean_returns, yerr=std_returns, fmt='o-', capsize=5, linewidth=2, markersize=8)
plt.xlabel('Number of Training Steps per Iteration')
plt.ylabel('Return')
plt.title(f'Effect of Training Steps on BC Performance ({env_name})')
plt.grid(True, alpha=0.3)

expert_mean = 3772.67
expert_std = 1.95
plt.axhline(y=expert_mean, color='r', linestyle='--', label=f'Expert Performance: {expert_mean:.2f}')
plt.fill_between([min(step_values), max(step_values)], 
                 expert_mean - expert_std, expert_mean + expert_std, 
                 color='r', alpha=0.1)

thirty_percent = 0.3 * expert_mean
plt.axhline(y=thirty_percent, color='g', linestyle='--', 
           label=f'30% of Expert Performance: {thirty_percent:.2f}')

plt.legend()
plt.tight_layout()

plt.savefig('hyperparameter_analysis.png', dpi=300)
plt.show()

print("\n" + "=" * 60)
print("Summary of results:")
print(f"Expert performance: {expert_mean:.2f} ± {expert_std:.2f}")
for i, steps in enumerate(step_values):
    print(f"{steps} steps: {mean_returns[i]:.2f} ± {std_returns[i]:.2f} " + 
          f"({mean_returns[i]/expert_mean*100:.1f}% of expert)")
    

plt.figure(figsize=(10, 6))
plt.errorbar(step_values, mean_returns, yerr=std_returns, fmt='o-', capsize=5, linewidth=2, markersize=8)
plt.xlabel('Number of Training Steps per Iteration')
plt.ylabel('Return')
plt.title(f'Effect of Training Steps on BC Performance ({env_name})')
plt.grid(True, alpha=0.3)

expert_mean = 3772.67
expert_std = 1.95
plt.axhline(y=expert_mean, color='r', linestyle='--', label=f'Expert Performance: {expert_mean:.2f}')
plt.fill_between([min(step_values), max(step_values)], 
                 expert_mean - expert_std, expert_mean + expert_std, 
                 color='r', alpha=0.1)

thirty_percent = 0.3 * expert_mean
plt.axhline(y=thirty_percent, color='g', linestyle='--', 
           label=f'30% of Expert Performance: {thirty_percent:.2f}')

for i, (x, y, std) in enumerate(zip(step_values, mean_returns, std_returns)):
    plt.annotate(f'σ = {std:.2f}', 
                 xy=(x, y), 
                 xytext=(0, 10),
                 textcoords='offset points',
                 ha='center', 
                 va='bottom',
                 fontsize=9,
                 bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))

plt.legend()
plt.tight_layout()