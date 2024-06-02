import copy
import os.path
import matplotlib.pyplot as plt
from utils import print_P
import numpy as np
import gym
from registration import register_env
register_env()


def value_iteration(env, gamma=0.99, epsilon=1e-6, max_iterations=10000):
    """
    Value Iteration algorithm to find the optimal value function.

    Parameters:
    - env: OpenAI Gym environment
    - gamma: Discount factor (default: 0.99)
    - epsilon: Convergence threshold (default: 1e-6)
    - max_iterations: Maximum number of iterations (default: 10000)

    Returns:
    - optimal_value_function: Optimal value function
    - optimal_policy: Optimal policy
    """
    num_states = env.observation_space.n
    num_actions = env.action_space.n

    # Initialize value function
    V = np.zeros(num_states)

    for iteration in range(max_iterations):
        print(f"itr = {iteration}")
        delta = 0
        prev_V = copy.deepcopy(V)
        # Update the value function for each state
        for s in range(num_states):
            if s == num_states - 2:
                print("Heya")
            # Calculate the Q-value for each action
            Q_values = [sum([p * (r + gamma * (1-int(t)) * prev_V[s_]) for p, s_, r, t in env.P[s][a]]) for a in range(num_actions)]
            # Q_values = [sum([p * (r + gamma * prev_V[s_]) for p, s_, r, t in env.P[s][a]]) for a in range(num_actions)]

            # Update the value function
            V[s] = max(Q_values)

            # Check for convergence
            delta = max(delta, np.abs(prev_V[s] - V[s]))

        # Check for convergence
        if delta < epsilon:
            break

    # Extract optimal policy from the optimal value function
    optimal_policy = np.zeros(num_states, dtype=int)
    for s in range(num_states):
        Q_values = [sum([p * (r + gamma * V[s_]) for p, s_, r, _ in env.P[s][a]]) for a in range(num_actions)]
        optimal_policy[s] = np.argmax(Q_values)

    return V, optimal_policy


if __name__ == "__main__":
    # Q-learning parameters
    gamma = 0.8  # Discount factor
    # gamma = 1.  # Discount factor

    # Create the Q-table
    env = gym.make('cliff_env')

    print_P(env)
    # env = gym.make('FrozenLake-v1', is_slippery=False)
    optimal_value_function, optimal_policy = value_iteration(env, gamma=gamma)

    from utils import reshape_V_func
    reshaped_opt_v = reshape_V_func(optimal_value_function, env)

    print("Optimal Value Function:")
    from matplotlib.colors import Normalize
    # print(optimal_value_function)
    pretty_v = np.array_str(reshaped_opt_v, precision=5, suppress_small=True)
    print(pretty_v)
    print("\nOptimal Policy:")
    print(optimal_policy)

    from utils import print_value

    print_value(reshaped_opt_v)

    from tmp_cliff import create_worst_case_env
    betas = 0.2 * np.ones((env.nS, env.nA))
    worst_env = create_worst_case_env(env, betas)
    # print_P(worst_env)
    optimal_robust_value_function, _ = value_iteration(worst_env, gamma=gamma)
    reshaped_opt_robust_v = reshape_V_func(optimal_robust_value_function, worst_env)
    print_value(reshaped_opt_robust_v)
