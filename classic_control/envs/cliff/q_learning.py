import os.path

import numpy as np
import gym
from envs.cliff.utils import print_P
from envs.cliff.registration import register_env
from envs.cliff.tmp_cliff import TERMINATE_WHEN_FALL, REG_REWARD, FALL_REWARD, SUCCESS_REWARD, CORRECT_PROB, REV_PROB, OTHER_PROB
import copy
import matplotlib.pyplot as plt
from envs.cliff.utils import set_seed
from pathlib import Path
import json


register_env()

SAVE_DIR = os.path.join(os.path.dirname(__file__), "q_learning")
LOAD = True


def non_robust_Q_learning(env, alpha=0.1, epsilon=0.1, gamma=0.99, num_episodes=1e5, optimal_v=None,
                          error_path=None, verbose=1):
    state_size = env.observation_space.n
    action_size = env.action_space.n
    # q_table = np.zeros((state_size, action_size))
    q_table = np.random.rand(state_size, action_size)
    errors = []
    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        while not done:
            # Exploration-exploitation trade-off
            if np.random.rand() < epsilon:
                action = env.action_space.sample()  # Explore
            else:
                action = np.argmax(q_table[state, :])  # Exploit

            # Take the chosen action and observe the next state and reward
            next_state, reward, terminated, truncated, info = env.step(action)
            done = truncated or terminated

            # Q-learning update rule
            q_table[state, action] = (1 - alpha) * q_table[state, action] + alpha * (reward + gamma * (1-int(done)) * np.max(q_table[next_state, :]))

            state = next_state

        if optimal_v is not None and verbose:
            print(f"episode {episode}\tQ_diff: {np.linalg.norm(np.max(q_table, axis=1) - optimal_v)}")
            errors.append(np.linalg.norm(np.max(q_table, axis=1) - optimal_v))

    if optimal_v is not None:
        plt.subplots()
        plt.plot(errors, label="")
        plt.xlabel("Timestep")
        plt.ylabel("Q-func error")
        if error_path is not None:
            plt.savefig(f"{error_path}.png")
            np.array(errors).dump(f"{error_path}.nd")
            plt.close()
        else:
            plt.show()
    return q_table


def robust_Q_learning(env, alpha=0.1, epsilon=0.1, gamma=0.99, num_episodes=1e5, optimal_v=None,
                          n_sample=5, kappa=0.2, error_path=None, verbose=1):
    state_size = env.observation_space.n
    action_size = env.action_space.n
    q_table = np.zeros((state_size, action_size))
    errors = []
    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        while not done:
            # Exploration-exploitation trade-off
            if np.random.rand() < epsilon:
                action = env.action_space.sample()  # Explore
            else:
                action = np.argmax(q_table[state, :])  # Exploit

            # sample n next_states from the env
            next_states_and_rewards = env.sample_n_next_steps(n=n_sample, action=action)
            next_rewards = [res[1] for res in next_states_and_rewards]
            next_states = [res[0] for res in next_states_and_rewards]
            next_states_val = q_table[next_states].max(axis=1)
            # next_states_weights = [r + gamma * q for (r, q) in zip(next_rewards, next_states_val)]
            next_states_weights = next_states_val

            # Compute \omega
            omega = np.mean(next_states_weights)
            normalized_weights = -1. * (next_states_weights - omega) / kappa
            e_weights = np.exp(normalized_weights)
            e_weights[e_weights == np.inf] = np.finfo(np.float32).max
            sum = np.sum(e_weights)
            if sum == np.inf:
                sum = np.finfo(np.float32).max
            final_weights = e_weights / sum
            if np.sum(final_weights) != 1:
                final_weights = final_weights / np.sum(final_weights)

            # Set new next state
            sampled_next_state_idx = np.random.choice(list(range(n_sample)), p=final_weights)
            sampled_next_state = next_states[sampled_next_state_idx]
            env.set_next_state(sampled_next_state, action_taken=action)

            # Take the chosen action and observe the next state and reward
            next_state, reward, terminated, truncated, info = env.step(action)
            done = truncated or terminated

            # Q-learning update rule
            q_table[state, action] = (1 - alpha) * q_table[state, action] + alpha * (
                        reward + gamma * (1-int(done)) * np.max(q_table[next_state, :]))

            state = next_state

        if optimal_v is not None and verbose:
            print(f"episode {episode}\tQ_diff: {np.linalg.norm(np.max(q_table, axis=1) - optimal_v)}")
            errors.append(np.linalg.norm(np.max(q_table, axis=1) - optimal_v))
    if optimal_v is not None:
        plt.plot(errors, label="")
        plt.xlabel("Timestep")
        plt.ylabel("Q-func error")
        if error_path is not None:
            plt.savefig(f"{error_path}.png")
            np.array(errors).dump(f"{error_path}.nd")
        else:
            plt.show()
    return q_table


if __name__ == "__main__":

    # Training parameters
    num_episodes = 20_000
    # num_episodes = 5_000
    error_epsilon = 1e-6
    alpha = 0.01  # Learning rate
    gamma = 0.8  # Discount factor
    # gamma = 1.  # Discount factor
    epsilon = 0.2  # Exploration-exploitation trade-off
    seed = 0

    # Uncertainty set
    BETA = 0.4
    n_sample = 5
    kappa = 0.4
    # Create the Q-table
    env = gym.make('cliff_env')

    # Set seed for reproducibility
    env.np_random = np.random.default_rng(seed)
    env.action_space.seed(seed)
    set_seed(seed)

    from v_iteration import value_iteration
    from utils import print_value, reshape_V_func, print_policy

    from tmp_cliff import create_worst_case_env, create_worst_case_env_2

    env_config = {
        "terminate_when_fall": TERMINATE_WHEN_FALL,
        "fall_reward": FALL_REWARD,
        "reg_reward": REG_REWARD,
        "success_reward": SUCCESS_REWARD,
        "correct_prob": CORRECT_PROB,
        "rev_prob": REV_PROB,
        "other_prob": OTHER_PROB
    }
    alg_config = {
        "env_config": env_config,
        "num_episodes": num_episodes,
        "error_epsilon": error_epsilon,
        "alpha": alpha,
        "gamma": gamma,
        "epsilon": epsilon,
        "seed": seed
    }
    env_text = (f"gamma_{gamma}_fall_{int(TERMINATE_WHEN_FALL)}_{FALL_REWARD}_succ_{SUCCESS_REWARD}_reg_{REG_REWARD}"
                f"_correct_{CORRECT_PROB}_rev_{REV_PROB}_other_{OTHER_PROB}")

    # -------------- Non-robust part --------------

    nominal_res_path = os.path.join(SAVE_DIR, env_text, "nominal", f"seed_{seed}")
    Path(nominal_res_path).mkdir(parents=True, exist_ok=True)

    env_path = os.path.join(nominal_res_path, "nonminal_env.png")
    print_P(env, file_path=env_path)

    # dump config to results directory
    config_path = os.path.join(nominal_res_path, "config.json")
    with open(config_path, "w") as outfile:
        json.dump(alg_config, outfile)

    optimal_v, optimal_policy = value_iteration(env, gamma=gamma)
    opt_v_func_path = os.path.join(nominal_res_path, "Optimal_v_func.png")
    reshaped_opt_v = reshape_V_func(optimal_v, env)
    opt_pi_path = os.path.join(nominal_res_path, "Optimal_pi.png")
    
    error_path = os.path.join(nominal_res_path, "error")
    q_func_path = os.path.join(nominal_res_path, "q_func.nd")
    if not LOAD:
        q_table = non_robust_Q_learning(env, alpha=alpha, epsilon=epsilon, gamma=gamma, optimal_v=optimal_v,
                                    num_episodes=num_episodes, error_path=error_path)
        # dump q_function to results directory
        with open(config_path, "w") as outfile:
            q_table.dump(q_func_path)
    else:
        q_table = np.load(q_func_path, allow_pickle=True)

    V = np.max(q_table, axis=1)
    reshaped_learned_v = reshape_V_func(V, env)
    learned_v_func_path = os.path.join(nominal_res_path, "Learned_v_func.png")
    learned_opt_pi_path = os.path.join(nominal_res_path, "Learned_pi.png")


    # -------------- Robust part --------------
    # Create log dir
    ewok_text = f"EWoK_beta_{BETA}_N_{n_sample}_kappa_{kappa}"
    robust_res_path = os.path.join(SAVE_DIR, env_text, ewok_text, f"seed_{seed}")
    Path(robust_res_path).mkdir(parents=True, exist_ok=True)

    env_path = os.path.join(robust_res_path, "nonminal_env.png")
    print_P(env, file_path=env_path)

    # dump config to results directory
    config_path = os.path.join(robust_res_path, "config.json")
    with open(config_path, "w") as outfile:
        json.dump(alg_config, outfile)

    betas = BETA * np.ones((env.nS, env.nA))
    # worst_env = create_worst_case_env(env, betas)
    worst_env = create_worst_case_env_2(env, betas)

    # Set seed for reproducibility
    worst_env.np_random = np.random.default_rng(seed)
    worst_env.action_space.seed(seed)
    set_seed(seed)

    worst_env_path = os.path.join(robust_res_path, "worst_env.png")
    print_P(worst_env, file_path=worst_env_path)

    optimal_robust_value_function, _ = value_iteration(worst_env, gamma=gamma)

    reshaped_opt_robust_v = reshape_V_func(optimal_robust_value_function, env)
    optimal_robust_v_func_path = os.path.join(robust_res_path, "Optimal_robust_v_func.png")
    opt_robust_pi_path = os.path.join(robust_res_path, "Optimal_pi.png")


    error_path = os.path.join(robust_res_path, "error")
    q_func_path = os.path.join(robust_res_path, "robust_q_func.nd")
    if not LOAD:
        robust_q_table = robust_Q_learning(env, alpha=alpha, epsilon=epsilon, gamma=gamma,
                                       optimal_v=optimal_robust_value_function,
                                       n_sample=n_sample, kappa=kappa, error_path=error_path,
                                       num_episodes=num_episodes)
        # dump q_function to results directory
        with open(config_path, "w") as outfile:
            robust_q_table.dump(q_func_path)
    else:
        robust_q_table = np.load(q_func_path, allow_pickle=True)

    V_robust = np.max(robust_q_table, axis=1)
    reshaped_robust_v = reshape_V_func(V_robust, env)
    learned_robust_v_func_path = os.path.join(robust_res_path, "Learned_robust_v_func.png")
    learned_robust_pi_path = os.path.join(robust_res_path, "Learned_pi.png")




    # ------ Printing part ---------------
    reshaped_opt_v_copy = copy.deepcopy(reshaped_opt_v).flatten()
    reshaped_opt_v_copy.sort()
    reshaped_learned_v_copy = copy.deepcopy(reshaped_learned_v).flatten()
    reshaped_learned_v_copy.sort()
    reshaped_opt_robust_v_copy = copy.deepcopy(reshaped_opt_robust_v).flatten()
    reshaped_opt_robust_v_copy.sort()
    reshaped_robust_v_copy = copy.deepcopy(reshaped_robust_v).flatten()
    reshaped_robust_v_copy.sort()
    min_val = min(reshaped_opt_v_copy[0], reshaped_learned_v_copy[0], reshaped_opt_robust_v_copy[0],
                  reshaped_robust_v_copy[0])
    max_val = max(reshaped_opt_v_copy[-2], reshaped_learned_v_copy[-2], reshaped_opt_robust_v_copy[-2],
                  reshaped_robust_v_copy[-2])
    scale = [min_val, max_val]
    # Print non-robust part
    print_value(reshaped_opt_v, file_path=opt_v_func_path, scale=scale)
    print_policy(reshaped_opt_v, file_path=opt_pi_path, scale=scale)
    print_value(reshaped_learned_v, file_path=learned_v_func_path, scale=scale)
    print_policy(reshaped_learned_v, file_path=learned_opt_pi_path, scale=scale)

    # Print robust_part
    print_value(reshaped_opt_robust_v, file_path=optimal_robust_v_func_path, scale=scale)
    print_policy(reshaped_opt_robust_v, file_path=opt_robust_pi_path, scale=scale)
    print_value(reshaped_robust_v, file_path=learned_robust_v_func_path, scale=scale)
    print_policy(reshaped_robust_v, file_path=learned_robust_pi_path, scale=scale)