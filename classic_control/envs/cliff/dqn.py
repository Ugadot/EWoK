import gym
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from envs.cliff.registration import register_env
from stable_baselines3.common.vec_env import DummyVecEnv
register_env()


# Create the Cliff Walking environment
env = DummyVecEnv([lambda: gym.make('cliff_env')])

# Define the DQN model
model = DQN("MlpPolicy", env, verbose=1)

# Train the model for 10000 episodes
model.learn(total_timesteps=100000)

import torch
import numpy as np
# Iterate over all possible states
V = np.zeros(env.observation_space.n)
for state in range(env.observation_space.n):
    q_values = model.q_net(torch.Tensor([state]).to(model.device)).cpu().detach().numpy()
    V[state] = np.max(q_values)

env = gym.make('cliff_env')
from utils import print_value, reshape_V_func
reshaped_opt_v = reshape_V_func(V, env)
print_value(reshaped_opt_v)