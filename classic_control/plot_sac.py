import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from rliable import library as rly
from rliable import metrics
from utils import get_colors

reds, blues, greens, oranges, purples, bones, greys, infernos, wistias = get_colors()

ALL_TASKS = ["walker_walk", "walker_stand", "walker_run"]

QUADRUPED_TEST_NOISES = np.linspace(-0.5, 0.5, 21)
WALKER_TEST_NOISES = np.linspace(-0.3, 0.3, 21)

TASKS_TEST_VALS = {
    "walker_walk": WALKER_TEST_NOISES,
    "walker_stand": WALKER_TEST_NOISES,
    "walker_run": WALKER_TEST_NOISES,
}

WALKER_TASK_PARAMS = {
    "thigh_length": np.linspace(0.1, 0.5, 21),
    "torso_length": np.linspace(0.1, 0.7, 21),
    "joint_damping": np.linspace(0.1, 10.0, 21),
}
WALKER_NOMINAL_VALS = {
    "thigh_length": 0.225,
    "torso_length": 0.3,
    "joint_damping": 0.1,
}

TASK_NOMINAL_VALS = {
    "walker_walk": WALKER_NOMINAL_VALS,
    "walker_stand": WALKER_NOMINAL_VALS,
    "walker_run": WALKER_NOMINAL_VALS,
}

TASK_ENV_PARAMS = {
    "walker_walk": WALKER_TASK_PARAMS,
    "walker_stand": WALKER_TASK_PARAMS,
    "walker_run": WALKER_TASK_PARAMS,
}

ALL_FONTS = ["Times New Roman", "Arial", "Helvetica", "Calibri"]
PLOT_FONT = ALL_FONTS[0]
ALPHA_LEVEL = 0.2
LINE_WIDTH = 5
MARKER_SIZE = 20
SCALE = 2.3
BORDER_STRECH_PERCENTAGE = 0.03
OUR_COLOR = blues[4]
BASELINE_COLOR = reds[4]
NOMINAL_COLOR = greys[4]
DR_COLOR = greens[4]
OUR_SHAPE = 's'
BASELINE_SHAPE = '^'
DR_SHAPE = 'o'

for task in ALL_TASKS:
    param_values = TASKS_TEST_VALS.get(task, None)
    fig_1 = plt.figure(figsize=(5*SCALE, 4*SCALE), dpi=300)
    plt.rcParams['font.family'] = PLOT_FONT

    # Collect results for baseline
    rewards = []
    folder = f"./continuous_results/sac/noise_change/{task}"
    runs = sorted(os.listdir(folder))
    for run in runs:
        rewards_per_run = []
        df = pd.read_csv(f"{folder}/{run}/eval.csv")
        for step, value in enumerate(param_values):
            reward = df[f'episode_reward ({step + 1}-th)'][1]
            rewards_per_run.append(reward)
        rewards.append(np.asarray(rewards_per_run))
    baseline_rewards = np.asarray(rewards)

    # Collect results for dr
    rewards = []
    folder = f"./continuous_results/sac/noise_change/{task}_dr"
    runs = sorted(os.listdir(folder))
    for run in runs:
        rewards_per_run = []
        df = pd.read_csv(f"{folder}/{run}/eval.csv")
        for step, value in enumerate(param_values):
            reward = df[f'episode_reward ({step + 1}-th)'][1]
            rewards_per_run.append(reward)
        rewards.append(np.asarray(rewards_per_run))
    dr_rewards = np.asarray(rewards)

    # Collect results for our method
    rewards = []
    folder = f"./continuous_results/sac/noise_change/{task}_reset"
    runs = sorted(os.listdir(folder))
    for run in runs:
        rewards_per_run = []
        df = pd.read_csv(f"{folder}/{run}/eval.csv")
        for step, value in enumerate(param_values):
            reward = df[f'episode_reward ({step + 1}-th)'][1]
            rewards_per_run.append(reward)
        rewards.append(np.asarray(rewards_per_run))
    our_rewards = np.asarray(rewards)

    # Calculate IQM and CIs
    our_iqm = []
    baseline_iqm = []
    dr_iqm = []
    for i in range(len(param_values)):
        result_dict = {"our": our_rewards[:, [i]], "baseline": baseline_rewards[:, [i]],
                       "dr": dr_rewards[:, [i]]}
        scores, score_cis = rly.get_interval_estimates(
            result_dict, metrics.aggregate_iqm, reps=10000)
        our_iqm.append((score_cis['our'][0, 0], scores['our'], score_cis['our'][1, 0]))
        baseline_iqm.append((score_cis['baseline'][0, 0], scores['baseline'], score_cis['baseline'][1, 0]))
        dr_iqm.append((score_cis['dr'][0, 0], scores['dr'], score_cis['dr'][1, 0]))
    our_iqm = np.asarray(our_iqm)
    baseline_iqm = np.asarray(baseline_iqm)
    dr_iqm = np.asarray(dr_iqm)

    min_y = np.inf
    max_y = -np.inf
    min_x = np.inf
    max_x = -np.inf

    # Plot nominal
    plt.axvline(x=0,  color=NOMINAL_COLOR, label='nominal value', linewidth=LINE_WIDTH, linestyle='--')

    # Plot baseline
    plt.plot(param_values, baseline_iqm[:, 1], f'{BASELINE_SHAPE}-', color=BASELINE_COLOR, label='SAC',
             markersize=MARKER_SIZE, linewidth=LINE_WIDTH)
    plt.fill_between(param_values, baseline_iqm[:, 0], baseline_iqm[:, 2],
                    linewidth=0, color=BASELINE_COLOR, alpha=ALPHA_LEVEL)
    min_y = min(min_y, min(baseline_iqm[:, 0]))
    max_y = max(max_y, max(baseline_iqm[:, 2]))
    min_x = min(param_values)
    max_x = max(param_values)

    # Plot dr
    plt.plot(param_values, dr_iqm[:, 1], f'{DR_SHAPE}-', color=DR_COLOR, label='dr',
             markersize=MARKER_SIZE, linewidth=LINE_WIDTH)
    plt.fill_between(param_values, dr_iqm[:, 0], dr_iqm[:, 2],
                    linewidth=0, color=DR_COLOR, alpha=ALPHA_LEVEL)
    min_y = min(min_y, min(dr_iqm[:, 0]))
    max_y = max(max_y, max(dr_iqm[:, 2]))
    min_x = min(param_values)
    max_x = max(param_values)

    # Plot ours
    plt.plot(param_values, our_iqm[:, 1], f'{OUR_SHAPE}-', color=OUR_COLOR, label='ours',
             markersize=MARKER_SIZE, linewidth=LINE_WIDTH)
    plt.fill_between(param_values, our_iqm[:, 0], our_iqm[:, 2],
                    linewidth=0, color=OUR_COLOR, alpha=ALPHA_LEVEL)
    min_y = min(min_y, min(our_iqm[:, 0]))
    max_y = max(max_y, max(our_iqm[:, 2]))
    min_x = min(param_values)
    max_x = max(param_values)

    save_path = f"./figs/sac/{task}"
    Path(save_path).mkdir(parents=True, exist_ok=True)

    # plt.legend()
    y_diff = max_y - min_y
    x_diff = max_x - min_x
    plt.ylim(min_y - BORDER_STRECH_PERCENTAGE * y_diff, max_y + BORDER_STRECH_PERCENTAGE * y_diff)
    plt.xlim(min_x - BORDER_STRECH_PERCENTAGE * x_diff, max_x + BORDER_STRECH_PERCENTAGE * x_diff)
    ax = fig_1.axes[0]
    plt.locator_params(axis='both', nbins=4)
    ax.xaxis.set_tick_params(width=3)
    ax.yaxis.set_tick_params(width=3)
    plt.tight_layout()
    plt.legend(fontsize="30")
    plt.savefig(f"{save_path}/noise")
    plt.close(fig_1)


    # Plot env params
    perturbations = TASK_ENV_PARAMS.get(task)
    for param, values in perturbations.items():
        fig_1 = plt.figure(figsize=(5*SCALE, 4*SCALE), dpi=300)
        plt.rcParams['font.family'] = PLOT_FONT

        # Collect results for baseline
        rewards = []
        folder = f"./continuous_results/sac/param_change/{task}_{param}"
        runs = sorted(os.listdir(folder))
        for run in runs[-10:]:
            df = pd.read_csv(f"{folder}/{run}/eval.csv")
            rewards.append(df['episode_reward'].to_numpy())
        baseline_rewards = np.asarray(rewards)

        # Collect results for dr
        rewards = []
        folder = f"./continuous_results/sac/param_change/{task}_dr_{param}"
        runs = sorted(os.listdir(folder))
        for run in runs:
            rewards_per_run = []
            df = pd.read_csv(f"{folder}/{run}/eval.csv")
            for step, value in enumerate(param_values):
                reward = df[f'episode_reward ({step + 1}-th)'][1]
                rewards_per_run.append(reward)
            rewards.append(np.asarray(rewards_per_run))
        dr_rewards = np.asarray(rewards)

        # Collect results for our method
        rewards = []
        folder = f"./continuous_results/sac/param_change/{task}_reset_{param}"
        runs = sorted(os.listdir(folder))
        for run in runs[-10:]:
            df = pd.read_csv(f"{folder}/{run}/eval.csv")
            rewards.append(df['episode_reward'].to_numpy())
        our_rewards = np.asarray(rewards)

        # Calculate IQM and CIs
        our_iqm = []
        baseline_iqm = []
        dr_iqm = []
        for i in range(len(param_values)):
            result_dict = {"our": our_rewards[:, [i]], "baseline": baseline_rewards[:, [i]],
                           "dr": dr_rewards[:, [i]]}
            scores, score_cis = rly.get_interval_estimates(
                result_dict, metrics.aggregate_iqm, reps=10000)
            our_iqm.append((score_cis['our'][0, 0], scores['our'], score_cis['our'][1, 0]))
            baseline_iqm.append((score_cis['baseline'][0, 0], scores['baseline'], score_cis['baseline'][1, 0]))
            dr_iqm.append((score_cis['dr'][0, 0], scores['dr'], score_cis['dr'][1, 0]))
        our_iqm = np.asarray(our_iqm)
        baseline_iqm = np.asarray(baseline_iqm)
        dr_iqm = np.asarray(dr_iqm)


        min_y = np.inf
        max_y = -np.inf
        min_x = np.inf
        max_x = -np.inf

        # Plot nominal
        nominal_val = TASK_NOMINAL_VALS[task][param]
        plt.axvline(x=nominal_val,  color=NOMINAL_COLOR, label='nominal value', linewidth=LINE_WIDTH, linestyle='--')

        # Plot baseline
        plt.plot(values, baseline_iqm[:, 1], f'{BASELINE_SHAPE}-', color=BASELINE_COLOR, label='SAC',
                 markersize=MARKER_SIZE, linewidth=LINE_WIDTH)
        plt.fill_between(values, baseline_iqm[:, 0], baseline_iqm[:, 2],
                        linewidth=0, color=BASELINE_COLOR, alpha=ALPHA_LEVEL)
        min_y = min(min_y, min(baseline_iqm[:, 0]))
        max_y = max(max_y, max(baseline_iqm[:, 2]))
        min_x = min(values)
        max_x = max(values)

        # Plot dr
        plt.plot(values, dr_iqm[:, 1], f'{DR_SHAPE}-', color=DR_COLOR, label='dr',
                 markersize=MARKER_SIZE, linewidth=LINE_WIDTH)
        plt.fill_between(values, dr_iqm[:, 0], dr_iqm[:, 2],
                         linewidth=0, color=DR_COLOR, alpha=ALPHA_LEVEL)
        min_y = min(min_y, min(dr_iqm[:, 0]))
        max_y = max(max_y, max(dr_iqm[:, 2]))
        min_x = min(values)
        max_x = max(values)

        # Plot ours
        plt.plot(values, our_iqm[:, 1], f'{OUR_SHAPE}-', color=OUR_COLOR, label='ours',
                 markersize=MARKER_SIZE, linewidth=LINE_WIDTH)
        plt.fill_between(values, our_iqm[:, 0], our_iqm[:, 2],
                        linewidth=0, color=OUR_COLOR, alpha=ALPHA_LEVEL)
        min_y = min(min_y, min(our_iqm[:, 0]))
        max_y = max(max_y, max(our_iqm[:, 2]))
        min_x = min(values)
        max_x = max(values)

        save_path = f"./figs/sac/{task}"
        Path(save_path).mkdir(parents=True, exist_ok=True)

        # plt.legend()
        y_diff = max_y - min_y
        x_diff = max_x - min_x
        plt.ylim(min_y - BORDER_STRECH_PERCENTAGE * y_diff, max_y + BORDER_STRECH_PERCENTAGE * y_diff)
        plt.xlim(min_x - BORDER_STRECH_PERCENTAGE * x_diff, max_x + BORDER_STRECH_PERCENTAGE * x_diff)
        ax = fig_1.axes[0]
        plt.locator_params(axis='both', nbins=4)
        ax.xaxis.set_tick_params(width=3)
        ax.yaxis.set_tick_params(width=3)
        plt.tight_layout()
        plt.legend(fontsize="30")
        plt.savefig(f"{save_path}/{param}")
        plt.close(fig_1)

