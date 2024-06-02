import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from rliable import library as rly
from rliable import metrics

from utils import get_colors

reds, blues, greens, oranges, purples, bones, greys, infernos, wistias = get_colors()


cmap = {0: greys[6], 0.5: reds[4], 1: wistias[4], 2: greens[4], 5: blues[4]}
smap = {0: '^', 0.5: 's', 1: 'D', 2: 'o', 5: '*'}

N_cmap = {0: greys[6], 5: reds[4], 10: wistias[4], 20: greens[4], 40: blues[4], 50: blues[4]}
N_smap = {0: '^', 5: 's', 10: 'D', 20: 'o', 40: '*', 50: '*'}

ALL_FONTS = ["Times New Roman", "Arial", "Helvetica", "Calibri"]
PLOT_FONT = ALL_FONTS[0]
ALPHA_LEVEL = 0.2
LINE_WIDTH = 5
MARKER_SIZE = 20
SCALE = 2.3
BORDER_STRECH_PERCENTAGE = 0.03

NOMINAL_COLOR = greys[4]

ALL_TASKS = ["walker_walk", "walker_stand", "walker_run"]
WALKER_TEST_NOISES = np.linspace(-0.3, 0.3, 21)

TASKS_TEST_VALS = {
    "walker_walk": WALKER_TEST_NOISES,
    "walker_stand": WALKER_TEST_NOISES,
    "walker_run": WALKER_TEST_NOISES,
}


for task in ALL_TASKS:
    param_values = TASKS_TEST_VALS.get(task, None)
    fig_1 = plt.figure(figsize=(5 * SCALE, 4 * SCALE), dpi=300)
    plt.rcParams['font.family'] = PLOT_FONT

    all_rewards = {}
    for temp in (0, 0.5, 1, 2, 5):
        # Collect results
        rewards = []
        folder = f"./continuous_results/sac/ablation/{task}/temp{temp}"
        runs = sorted(os.listdir(folder))
        for run in runs:
            rewards_per_run = []
            df = pd.read_csv(f"{folder}/{run}/eval.csv")
            for step, value in enumerate(param_values):
                reward = df[f'episode_reward ({step + 1}-th)'][1]
                rewards_per_run.append(reward)
            rewards.append(np.asarray(rewards_per_run))
        all_rewards[temp] = np.asarray(rewards)

    # Calculate IQM and CIs
    scores4plot = []
    for i in range(len(param_values)):
        result_dict = {k: v[:, [i]] for k, v in all_rewards.items()}
        scores, score_cis = rly.get_interval_estimates(
            result_dict, metrics.aggregate_iqm, reps=10000)
        scores4plot.append({k: (score_cis[k][0, 0], scores[k], score_cis[k][1, 0]) for k in scores})

    min_y = np.inf
    max_y = -np.inf
    min_x = np.inf
    max_x = -np.inf

    # Plot nominal
    plt.axvline(x=0,  color=NOMINAL_COLOR, label='nominal value', linewidth=LINE_WIDTH, linestyle='--')

    # Plot each temp results
    for key in all_rewards:
        y = np.array([score4plot[key][1] for score4plot in scores4plot])
        y_low = np.array([score4plot[key][0] for score4plot in scores4plot])
        y_high = np.array([score4plot[key][2] for score4plot in scores4plot])

        plt.plot(param_values, y, f'{smap[key]}-', color=cmap[key], label=f'temp={key}',
                 markersize=MARKER_SIZE, linewidth=LINE_WIDTH)
        plt.fill_between(param_values,  y_low, y_high,
                        linewidth=0, color=cmap[key], alpha=ALPHA_LEVEL)

        min_y = min(min_y, min(y_low))
        max_y = max(max_y, max(y_high))
        min_x = min(param_values)
        max_x = max(param_values)

    save_path = f"./figs/sac/ablation/kappa"

    Path(save_path).mkdir(parents=True, exist_ok=True)

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
    plt.title(f"{task}", fontsize=30)
    plt.savefig(f"{save_path}/{task}")
    plt.close(fig_1)


    # Plot IQM difference
    fig_1 = plt.figure(figsize=(5 * SCALE, 4 * SCALE), dpi=300)
    plt.rcParams['font.family'] = PLOT_FONT
    # Plot nominal
    plt.axvline(x=0, color=NOMINAL_COLOR, label='nominal value', linewidth=LINE_WIDTH, linestyle='--')
    # Plot zero diff line
    plt.axhline(y=0, color='k', linestyle='-', linewidth=1)

    min_y = np.inf
    max_y = -np.inf
    min_x = np.inf
    max_x = -np.inf

    iqm_baselines = np.array([score4plot[0][1] for score4plot in scores4plot])
    for key in all_rewards:
        if key != 0:
            iqm_key = np.array([score4plot[key][1] for score4plot in scores4plot])
            iqm_diff = iqm_key - iqm_baselines

            plt.plot(param_values, iqm_diff, f'{smap[key]}-', color=cmap[key], label=f'temp={key}',
                     markersize=MARKER_SIZE, linewidth=LINE_WIDTH)

            min_y = min(min_y, min(iqm_diff))
            max_y = max(max_y, max(iqm_diff))
            min_x = min(param_values)
            max_x = max(param_values)

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
    plt.title(f"{task}_diff)iqm", fontsize=30)
    plt.savefig(f"{save_path}/iqm_diff_{task}")
    plt.close(fig_1)

##################################################################################

    # Plot the same for size ablation
    param_values = TASKS_TEST_VALS.get(task, None)
    fig_1 = plt.figure(figsize=(5 * SCALE, 4 * SCALE), dpi=300)
    plt.rcParams['font.family'] = PLOT_FONT

    if task.startswith("walker"):
        param_values = np.linspace(-0.3, 0.3, 21)
        Ns = (0, 5, 10, 20, 50)
    else:
        param_values = np.linspace(-0.5, 0.5, 21)
        Ns = (0, 5, 10, 20, 40)

    all_rewards = {}
    for n_sample in Ns:
        # Collect results
        rewards = []
        folder = f"./continuous_results/sac/ablation/{task}/N{n_sample}"
        runs = sorted(os.listdir(folder))
        for run in runs:
            rewards_per_run = []
            df = pd.read_csv(f"{folder}/{run}/eval.csv")
            for step, value in enumerate(param_values):
                reward = df[f'episode_reward ({step + 1}-th)'][1]
                rewards_per_run.append(reward)
            rewards.append(np.asarray(rewards_per_run))
        all_rewards[n_sample] = np.asarray(rewards)

    # Calculate IQM and CIs
    scores4plot = []
    for i in range(len(param_values)):
        result_dict = {k: v[:, [i]] for k, v in all_rewards.items()}
        scores, score_cis = rly.get_interval_estimates(
            result_dict, metrics.aggregate_iqm, reps=10000)
        scores4plot.append({k: (score_cis[k][0, 0], scores[k], score_cis[k][1, 0]) for k in scores})

    min_y = np.inf
    max_y = -np.inf
    min_x = np.inf
    max_x = -np.inf

    # Plot nominal
    plt.axvline(x=0, color=NOMINAL_COLOR, label='nominal value', linewidth=LINE_WIDTH, linestyle='--')

    # Plot each temp results
    for key in all_rewards:
        y = np.array([score4plot[key][1] for score4plot in scores4plot])
        y_low = np.array([score4plot[key][0] for score4plot in scores4plot])
        y_high = np.array([score4plot[key][2] for score4plot in scores4plot])

        plt.plot(param_values, y, f'{N_smap[key]}-', color=N_cmap[key], label=f'temp={key}',
                 markersize=MARKER_SIZE, linewidth=LINE_WIDTH)
        plt.fill_between(param_values, y_low, y_high,
                         linewidth=0, color=N_cmap[key], alpha=ALPHA_LEVEL)

        min_y = min(min_y, min(y_low))
        max_y = max(max_y, max(y_high))
        min_x = min(param_values)
        max_x = max(param_values)

    save_path = f"./figs/sac/ablation/n_sample"
    Path(save_path).mkdir(parents=True, exist_ok=True)

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
    plt.title(f"{task}", fontsize=30)
    plt.savefig(f"{save_path}/{task}")
    plt.close(fig_1)

    # Plot IQM difference
    fig_1 = plt.figure(figsize=(5 * SCALE, 4 * SCALE), dpi=300)
    plt.rcParams['font.family'] = PLOT_FONT
    # Plot nominal
    plt.axvline(x=0, color=NOMINAL_COLOR, label='nominal value', linewidth=LINE_WIDTH, linestyle='--')
    # Plot zero diff line
    plt.axhline(y=0, color='k', linestyle='-', linewidth=1)

    min_y = np.inf
    max_y = -np.inf
    min_x = np.inf
    max_x = -np.inf

    iqm_baselines = np.array([score4plot[0][1] for score4plot in scores4plot])
    for key in all_rewards:
        if key != 0:
            iqm_key = np.array([score4plot[key][1] for score4plot in scores4plot])
            iqm_diff = iqm_key - iqm_baselines

            plt.plot(param_values, iqm_diff, f'{N_smap[key]}-', color=N_cmap[key], label=f'temp={key}',
                     markersize=MARKER_SIZE, linewidth=LINE_WIDTH)

            min_y = min(min_y, min(iqm_diff))
            max_y = max(max_y, max(iqm_diff))
            min_x = min(param_values)
            max_x = max(param_values)

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
    plt.title(f"{task}_diff)iqm", fontsize=30)
    plt.savefig(f"{save_path}/iqm_diff_{task}")
    plt.close(fig_1)