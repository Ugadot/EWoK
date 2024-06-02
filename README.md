# Bring Your Own (non-robust) Algorithm to Solve Robust MDPs by Estimating The Worst Kernel
## Documentation

### Abstract
Robust Markov Decision Processes (RMDPs) provide a framework for sequential decision-making that is robust to perturbations on the transition kernel.
However, current RMDP methods are often limited to small-scale problems, hindering their use in high-dimensional domains. 
To bridge this gap, we present **EWoK**, 
a novel online approach to solve RMDP that **E**stimates the **Wo**rst transition **K**ernel to learn robust policies.
Unlike previous works that regularize the policy or value updates, EWoK achieves robustness by simulating the worst scenarios for the agent while retaining complete flexibility in the learning process.
Notably, EWoK can be applied on top of any off-the-shelf _non-robust_ RL algorithm, enabling easy scaling to high-dimensional domains.
Our experiments, spanning from simple Cartpole to high-dimensional DeepMind Control Suite environments, demonstrate the effectiveness and applicability of the EWoK paradigm as a practical method for learning robust policies.
### Paper
Check out our paper at TODO

[//]: # ([Project Book]&#40;doc/RL_Project_Book.pdf&#41;)


## Get Started

### Requirements
1. python 3.7 (or newer)
2. install requirements:
```
    pip install -r requirements.txt
```
### Using the code

#### Train DQN agent for classic control environment Cartpole

The noisy version of OpenAI's gym environment: Cartpole is implemented in [classic_control](envs/classic_control).
We allow multiple next_state sampling in the environment in order to implement the Adversarial Kernel Approximation.

You can either train a single agent using the `run_classic_control.py`.
Or run train several agents using the python script in `scripts/script_classic_control.py`.

You can log your results to Weights & Biases, or use the pickled version that would be saved in `./logs` directory.


#### Train PPO agent for classic control environment Cartpole
You can either train a single agent using the `run_classic_control_PPO.py`.
Or run train several agents using the python script in `scripts/script_classic_control.py`. with the flag `--ppo`

You can log your results to Weights & Biases, or use the pickled version that would be saved in `./logs` directory.

For further information on all flags of this scripts run the following command:
```
python scripts/script_classic_control.py --help
```


### Exporting results to matplotlib graphs
You can also use the python scripts `draw_cartpole_DQN.py` and `draw_cartpole_ppo.py`. in order to draw an IQM graph of the results.
It is using the logs of the run based on the nominal values inserted in the script.

