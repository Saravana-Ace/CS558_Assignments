import subprocess
import time
import os

environments = [
    'Ant-v4', 
    'Hopper-v4'
    # 'HalfCheetah-v4',
    # 'Walker2d-v4' 
]

print("Starting behavior cloning experiments for all environments...")

for env_name in environments:
    print(f"\n\n{'='*50}")
    print(f"Running behavior cloning on {env_name}")
    print(f"{'='*50}\n")
    
    cmd = [
        'python', 'cs558/scripts/run_hw2.py',
        '--expert_policy_file', f'cs558/policies/experts/{env_name.split("-")[0]}.pkl',
        '--env_name', env_name,
        '--exp_name', f'bc_{env_name.lower().split("-")[0]}',
        '--n_iter', '1',
        '--expert_data', f'cs558/expert_data/expert_data_{env_name}.pkl',
        '--video_log_freq', '-1',
        '--eval_batch_size', '5000',
        # '--train_batch_size', '2000',
        # '--num_agent_train_steps_per_iter', '1500'
    ]
    
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output.strip())

print("\n\nAll experiments completed!")
print("You can analyze the results in the 'data' directory or using tensorboard:")
print("python -m tensorboard.main --logdir data")