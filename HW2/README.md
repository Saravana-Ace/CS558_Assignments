## Some potentially useful links for pytorch and gym
- For pytorch:  
	- https://www.cs.utexas.edu/~yukez/cs391r_fall2021/slides/tutorial_09-29_pytorch_intro.pdf#page=6.00  
	- https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html  
	- https://rail.eecs.berkeley.edu/deeprlcourse/deeprlcourse/static/slides/lec-3.pdf#page=25.00  
	- https://cs230.stanford.edu/blog/pytorch/  
- For openai gym:  
	- https://www.gocoder.one/blog/rl-tutorial-with-openai-gym/  
	- https://www.gymlibrary.dev/content/tutorials/  
	- https://blog.paperspace.com/getting-started-with-openai-gym/  


## Setup

You can run this code on your own machine or on Google Colab. We recommend Python 3.8+.  

**Local option:** If you choose to run locally, you will need to install MuJoCo and some Python packages; see [installation.md](installation.md) for instructions.


## Complete the code

Fill in sections marked with `TODO`. In particular, see
 - [infrastructure/rl_trainer.py](cs558/infrastructure/rl_trainer.py)
 - [policies/MLP_policy.py](cs558/policies/MLP_policy.py)
 - [infrastructure/replay_buffer.py](cs558/infrastructure/replay_buffer.py)
 - [infrastructure/utils.py](cs558/infrastructure/utils.py)
 - [infrastructure/pytorch_util.py](cs558/infrastructure/pytorch_util.py)

Look for sections maked with `HW2` to see how the edits you make will be used.
Some other files that you may find relevant
 - [scripts/run_hw2.py](cs558/scripts/run_hw2.py)
 - [agents/bc_agent.py](cs558/agents/bc_agent.py)

See the homework pdf for more details.

## Run the code

Tip: While debugging, you probably want to keep the flag `--video_log_freq -1` which will disable video logging and speed up the experiment. However, feel free to remove it to save videos of your awesome policy!

### Section 1 (Behavior Cloning)
Command for problem 1:

```
python cs558/scripts/run_hw2.py \
	--expert_policy_file cs558/policies/experts/Ant.pkl \
	--env_name Ant-v4 --exp_name bc_ant --n_iter 1 \
	--expert_data cs558/expert_data/expert_data_Ant-v4.pkl \
	--video_log_freq -1
```

Make sure to also try another environment.
See the homework PDF for more details on what else you need to run.
To generate videos of the policy, remove the `--video_log_freq -1` flag.

### Section 2 (DAgger)
Command for section 1:
(Note the `--do_dagger` flag, and the higher value for `n_iter`)

```
python cs558/scripts/run_hw2.py \
    --expert_policy_file cs558/policies/experts/Ant.pkl \
    --env_name Ant-v4 --exp_name dagger_ant --n_iter 10 \
    --do_dagger --expert_data cs558/expert_data/expert_data_Ant-v4.pkl \
	--video_log_freq -1
```

Make sure to also try another environment.
See the homework PDF for more details on what else you need to run.

## Visualization the saved tensorboard event file:

You can visualize your runs using tensorboard:
```
python -m tensorboard.main --logdir data
```

You will see scalar summaries as well as videos of your trained policies (in the 'images' tab).

You can choose to visualize specific runs with a comma-separated list:
```
python -m tensorboard.main --logdir data/run1,data/run2,data/run3...
```

## Credit:

This homework is directly taken from Guanya Shi's CMU 16-831 Introduction to Robot Learning class, with slight modifications.
Reference: https://16-831-s24.github.io/