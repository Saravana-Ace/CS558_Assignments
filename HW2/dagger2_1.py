import subprocess
import numpy as np
import matplotlib.pyplot as plt
import re
import os
import json

environments = ["Ant-v4", "Hopper-v4"]

n_iter = 10

results = {}

for env_name in environments:
    print(f"\n\nRunning DAgger on {env_name}")
    print("=" * 60)    

    cmd = [
        'python', 'cs558/scripts/run_hw2.py',
        '--expert_policy_file', f'cs558/policies/experts/{env_name.split("-")[0]}.pkl',
        '--env_name', env_name,
        '--exp_name', f'dagger_{env_name.lower().split("-")[0]}',
        '--n_iter', str(n_iter),
        '--do_dagger',
        '--expert_data', f'cs558/expert_data/expert_data_{env_name}.pkl',
        '--video_log_freq', '-1',
        '--train_batch_size', '100',
        '--eval_batch_size', '5000'
    ]
    
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    output, error = process.communicate()
    
    iteration_data = []
    
    expert_match = re.search(r'Initial_DataCollection_AverageReturn : ([\d\.\-]+)', output)
    if expert_match:
        expert_performance = float(expert_match.group(1))
    else:
        expert_performance = None
    
    iter_matches = re.finditer(r'\*+ Iteration (\d+) \*+', output)
    for iter_match in iter_matches:
        iter_num = int(iter_match.group(1))
        
        start_pos = iter_match.end()
        next_iter_match = re.search(r'\*+ Iteration \d+ \*+', output[start_pos:])
        end_pos = start_pos + (next_iter_match.start() if next_iter_match else len(output[start_pos:]))
        
        mean_match = re.search(r'Eval_AverageReturn : ([\d\.\-]+)', output[start_pos:end_pos])
        std_match = re.search(r'Eval_StdReturn : ([\d\.\-]+)', output[start_pos:end_pos])
        
        if mean_match and std_match:
            mean = float(mean_match.group(1))
            std = float(std_match.group(1))
            iteration_data.append((iter_num, mean, std))
            print(f"Iteration {iter_num}: Mean return = {mean:.2f}, Std = {std:.2f}")
    
    results[env_name] = {
        "expert_performance": expert_performance,
        "iterations": iteration_data
    }

with open('dagger_results.json', 'w') as f:
    json.dump(results, f)

for env_name, env_results in results.items():
    plt.figure(figsize=(10, 6))
    
    iterations = [i[0] for i in env_results["iterations"]]
    means = [i[1] for i in env_results["iterations"]]
    stds = [i[2] for i in env_results["iterations"]]
    
    plt.errorbar(iterations, means, yerr=stds, fmt='o-', capsize=5, 
                 linewidth=2, markersize=8, label='DAgger')
    
    expert_perf = env_results["expert_performance"]
    plt.axhline(y=expert_perf, color='r', linestyle='--', 
                label=f'Expert Performance: {expert_perf:.2f}')
    
    bc_perf = means[0]
    bc_std = stds[0]
    plt.axhline(y=bc_perf, color='g', linestyle='--', 
                label=f'BC Performance: {bc_perf:.2f}')
    
    thirty_percent = 0.3 * expert_perf
    plt.axhline(y=thirty_percent, color='purple', linestyle=':', 
               label=f'30% of Expert: {thirty_percent:.2f}')
    
    plt.xlabel('DAgger Iterations')
    plt.ylabel('Return')
    plt.title(f'DAgger Learning Curve ({env_name})')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(f'dagger_learning_curve_{env_name.split("-")[0]}.png', dpi=300)

print("\nPlots saved as dagger_learning_curve_Ant.png and dagger_learning_curve_Hopper.png")