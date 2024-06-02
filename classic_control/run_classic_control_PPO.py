from stable_baselines3.ppo import PPO
from sb3_extra.ewok_ppo import EWoKPPO
from sb3_extra.dmn_rand_ppo import DMNRANDPPO

import gymnasium as gym
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from stable_baselines3.common.utils import set_random_seed
from wandb.integration.sb3 import WandbCallback
import wandb
import numpy as np
import torch as th
from pathlib import Path
import pickle
import argparse
from envs.classic_control.registration import register_envs
import copy
import re
from envs.utils import get_update_env_params


# Nominal values
# Cartpole
X_NOISE_MEAN = 0.01
X_NOISE_STD = 0.01
THETA_NOISE_MEAN = 0.01
THETA_NOISE_STD = 0.01
POLE_LEN = 0.5
CART_MASS = 1.0
POLE_MASS = 0.1
GRAVITY_CP = 9.8

ENV_N_TIMESTEPS = {
    "Cartpole": 100_000,
}

ENV_NAMES = ["Cartpole"]

ENV_ALGO_KWARGS = {
    "Cartpole":
        {
            "batch_size": 32,
            "n_steps": 32,
            "gae_lambda": 0.8,
            "n_epochs": 20,
            "gamma": 0.98,
            "ent_coef": 0.0,
            "learning_rate": 0.001,
            "clip_range": 0.2,
         },
}

ENV_NOMINAL_VALS = {
    "Cartpole":
        {
            "x_noise_mean": X_NOISE_MEAN,
            "x_noise_std": X_NOISE_STD,
            "theta_noise_mean": THETA_NOISE_MEAN,
            "theta_noise_std": THETA_NOISE_STD,
            "pole_len": POLE_LEN,
            "gravity":  GRAVITY_CP,
            "pole_mass": POLE_MASS,
            "cart_mass": CART_MASS
        },

}

ENV_TEST_PARAMS = {
    "Cartpole":
        {
            "x_noise_std": [a * (0.1 / 20) for a in list(range(0, 21))],
            "x_noise_mean": np.linspace(-0.05, 0.05, 21),
            "theta_noise_std": [a * (0.05 / 20) for a in list(range(0, 21))],
            "theta_noise_mean": np.linspace(-0.05, 0.05, 21),
            "pole_len": [a * (3 / 20) for a in list(range(1, 21))],
            "gravity": [0.1, 0.3, 0.5, 0.7, 1, 5, 10, 15, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
            "pole_mass": [a * (5 / 20) for a in list(range(1, 21))],
            "cart_mass": [a * (5 / 20) for a in list(range(1, 21))]
        },
}

DMN_RAND_KEYS = [ENV_TEST_PARAMS[env_name].keys() for env_name in ENV_TEST_PARAMS.keys()]
DMN_RAND_KEYS = [key for env_keys in DMN_RAND_KEYS for key in env_keys]

register_envs(ENV_NAMES)


def main(args):
    set_random_seed(args.train_seed, th.cuda.is_available())
    env_nominal_kwargs = ENV_NOMINAL_VALS[args.env_name]

    env_noise_keys = [key for key in env_nominal_kwargs.keys() if re.search("noise", key)]
    # Change env_nominal_noises
    if len(args.nominal_noises) != len(env_noise_keys):
        raise ValueError(f"Gave wrong number of noises expected:\t{' '.join(env_noise_keys)}")
    else:
        for idx, key in enumerate(env_noise_keys):
            env_nominal_kwargs[key] = args.nominal_noises[idx]
            print(f"Set env {key}={args.nominal_noises[idx]}")

    env = gym.make(f"Noisy_{args.env_name}-v1", seed=args.train_seed, **env_nominal_kwargs)
    eval_env = gym.make(f"Noisy_{args.env_name}-v1", seed=args.master_test_seed, **env_nominal_kwargs)

    env.reset(seed=args.train_seed)
    eval_env.reset(seed=args.master_test_seed)

    algo_kwargs = {}
    if args.algo == "PPO":
        algo = PPO
        algo_text = args.algo
    elif args.algo == "EWoKPPO":
        algo = EWoKPPO
        algo_kwargs["n_sample"] = args.n_sample
        algo_kwargs["kappa"] = args.kappa
        algo_text = f"{args.algo}_n_sample_{args.n_sample}_kappa_{args.kappa}"
    elif args.algo == "PPO_Domain_Rand":
        algo = DMNRANDPPO
        # Create env_param range
        param_range_dict = {}
        env_test_params = ENV_TEST_PARAMS[args.env_name]
        for test_key, test_list in env_test_params.items():
            if test_key in args.dmn_rand_keys:
                param_range_dict[test_key] = {"is_discrete": False,
                                              "range": test_list}
        update_func = get_update_env_params(param_range_dict)
        algo_kwargs["update_env_params_func"] = update_func
        algo_text = f"{args.algo}_{args.dmn_rand_name}"
    else:
        raise ValueError("unrecognized algorithm")

    nominal_vals_text = "_".join([f"{k}_{env_nominal_kwargs[k]}" for k in env_noise_keys])

    model_dir = f"models/{args.env_name}/{nominal_vals_text}/{algo_text}"
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    model_path = f"{model_dir}/seed_{args.train_seed}"

    if args.use_wandb:
        wandb_name = f"{algo_text}_seed_{args.train_seed}"
        if args.load:
            wandb_name = wandb_name+"_LOADED"

        wandb.login(key=args.wandb_key, relogin=True)
        wandb_logger = wandb.init(entity=args.wandb_entity,
                                project=f"{args.env_name}_{nominal_vals_text}",
                                group=f"{algo_text}",
                                name=wandb_name,
                                sync_tensorboard=True,
                                save_code=False,)

    if args.load is False:
        env_algo_kwargs = ENV_ALGO_KWARGS[args.env_name]
        model = algo("MlpPolicy",
                     env,
                     verbose=1,
                     tensorboard_log=f"tb_log/{args.env_name}/{nominal_vals_text}/{algo_text}/seed_{args.train_seed}",
                     seed=args.train_seed,
                     **env_algo_kwargs,
                     **algo_kwargs
                     )

        eval_callback = EvalCallback(eval_env, best_model_save_path=f"{model_path}", n_eval_episodes=20,
                                     log_path=f"./logs/eval/{args.env_name}/{nominal_vals_text}/{algo_text}/seed_{args.train_seed}",
                                     eval_freq=1_000,
                                     deterministic=True, render=False)
        if args.use_wandb:
            wandb_callback = WandbCallback(
                            verbose=2)
            callback = CallbackList([wandb_callback, eval_callback])
        else:
            callback = eval_callback

        time_steps = ENV_N_TIMESTEPS[args.env_name]
        model.learn(total_timesteps=time_steps,
                    log_interval=4,
                    callback=callback)
        del model

    log_path = f"logs/{args.env_name}/{nominal_vals_text}/{algo_text}/seed_{args.train_seed}"
    Path(log_path).mkdir(parents=True, exist_ok=True)

    # Loading model and testing it
    model = algo.load(path=f"{model_path}/best_model.zip")

    np.random.seed(args.master_test_seed)
    test_seeds = np.random.randint(0, 2 ** 32 - 1, args.test_episodes)
    test_seeds = [a.item() for a in test_seeds]

    # TODO: Add this if needed
    env_test_params = ENV_TEST_PARAMS[args.env_name]
    for test_key, test_list in env_test_params.items():
        results = []
        # Eval
        for val in test_list:
            rewards = []
            for seed in test_seeds:
                set_random_seed(seed, using_cuda=th.cuda.is_available())
                kwargs = copy.deepcopy(env_nominal_kwargs)
                kwargs[f'{test_key}'] = val
                env = gym.make(f"Noisy_{args.env_name}-v1", seed=args.train_seed, **kwargs)

                obs, _ = env.reset(seed=seed)
                final_reward = 0
                done = False
                while not done:
                    action, _states = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    final_reward += reward
                rewards.append(final_reward)
            mean_reward = np.mean(rewards)
            results.append(mean_reward)
            if args.use_wandb:
                wandb_logger.log({f"test/{test_key}": val,
                              f"test/{test_key}_avg_reward": mean_reward})

        data = [[x, y] for (x, y) in zip(test_list, results)]
        with open(f"{log_path}/{test_key}.pkl", "wb") as fp:
            pickle.dump(data, fp)
    if args.use_wandb:
        wandb_logger.finish()


def get_params():
    parser = argparse.ArgumentParser(
        prog='ProgramName',
        description='What the program does',
        epilog='Text at the bottom of help')

    parser.add_argument('-env_name', '--env_name', type=str, choices=['Cartpole'])
    parser.add_argument('-a', '--algo', type=str, choices=['PPO', 'EWoKPPO', 'PPO_Domain_Rand'])
    parser.add_argument('-n_sample', '--n_sample', type=int, default=5)
    parser.add_argument('-k', '--kappa', type=float, default=1)
    parser.add_argument('-load', '--load', action='store_true', default=False)  # on/off flag
    parser.add_argument('-nominal_noises', '--nominal_noises', type=float, default=[0, 0.01, 0, 0.01], nargs="*")
    parser.add_argument('-test_episodes', '--test_episodes', type=int, default=30)
    parser.add_argument('-train_seed', '--train_seed', type=int, default=1)
    parser.add_argument('-master_test_seed', '--master_test_seed', type=int, default=321)

    parser.add_argument('-dmn_rand_keys', '--dmn_rand_keys', type=str, nargs="*", choices=DMN_RAND_KEYS)
    parser.add_argument('-dmn_rand_name', '--dmn_rand_name', type=str, default="NoName")
    parser.add_argument('-use_wandb', '--use_wandb', action='store_true', default=False)
    parser.add_argument('-wandb_key', '--wandb_key', type=str)
    parser.add_argument('-wandb_entity', '--wandb_entity', type=str)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_params()
    main(args)
