import pickle
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import re
import os
from rliable import metrics
from rliable import library as rly

from utils import get_colors

# Nominal values
# Cartpole
X_NOISE_MEAN = 0
X_NOISE_STD = 0.01
THETA_NOISE_MEAN = 0
THETA_NOISE_STD = 0.01
POLE_LEN = 0.5
CART_MASS = 1.0
POLE_MASS = 0.1
GRAVITY = 9.8

ALL_ENV_NAMES = ["Cartpole"]
fields = ["all_noise", "cart_mass", "pole_mass", "pole_len", "gravity"]

ENV_NOMINAL_VALS = {
    "Cartpole":
        {
            "x_noise_mean": X_NOISE_MEAN,
            "x_noise_std": X_NOISE_STD,
            "theta_noise_mean": THETA_NOISE_MEAN,
            "theta_noise_std": THETA_NOISE_STD,
            "pole_len": POLE_LEN,
            "gravity": GRAVITY,
            "pole_mass": POLE_MASS,
            "cart_mass": CART_MASS
        },
}

ENV_NOMINAL_NOISES = {
    "Cartpole": [0.0, 0.01, 0.0, 0.01],
}

reds, blues, greens, oranges, purples, bones, greys, infernos, wistias = get_colors()

OUR_COLOR = blues[4]
BASELINE_COLOR = reds[4]
NOMINAL_COLOR = greys[4]
OUR_SHAPE = 's'
BASELINE_SHAPE = '^'

ALPHA_LEVEL = 0.2
LINE_WIDTH = 5
MARKER_SIZE = 20
SCALE = 2.3
BORDER_STRECH_PERCENTAGE = 0.03

cmap = {0: greys[6], 3: reds[4], 6: wistias[4], 9: greens[4], 15: blues[4]}
smap = {0: '^', 3: 's', 6: 'D', 9: 'o', 15: '*'}

ENV_OUR_CONFIG = {
    "Cartpole":
        [
        {"ALGO": "EWoKPPO",
         "N_SAMPLE": 15,
         "KAPPA": 0.2,
         "label": "Ours",
         "color": cmap[15],
         "shape": smap[3]},
         {"ALGO": f"Domain_Rand",
          "label": "Domain_Rand",
          "color": cmap[9],
          "shape": smap[9]},
         ],
}

BASE_CONFIG = {"ALGO": "PPO",
            "label": "PPO",
            "color": BASELINE_COLOR,
            "shape": BASELINE_SHAPE}

DATA_FIELDS ={
    "Cartpole": {
        "pole_length": {
            "pkl_name": "pole_len",
            "x_axis": "pole len",
            "title": "Reward for different pole lengths",
            "nominal_val": POLE_LEN,
            "use_log_scale": False
        },
        "pole_mass": {
            "pkl_name": "pole_mass",
            "x_axis": "pole mass",
            "title": "Reward for different pole masses",
            "nominal_val": POLE_MASS,
            "use_log_scale": False
        },
        "gravity": {
            "pkl_name": "gravity",
            "x_axis": "max gravity",
            "title": "Reward for different max gravities",
            "nominal_val": GRAVITY,
            "use_log_scale": False
        },
        "cart_mass": {
            "pkl_name": "cart_mass",
            "x_axis": "cart mass",
            "title": "Reward for different cart masses",
            "nominal_val": CART_MASS,
            "use_log_scale": False
        },
        "theta_noise_std": {
            "pkl_name": "theta_noise_std",
            "x_axis": "theta_noise_std",
            "title": "Reward for different theta_noise_std",
            "nominal_val": THETA_NOISE_STD,
            "use_log_scale": False
        },
        "x_noise_std": {
            "pkl_name": "x_noise_std",
            "x_axis": "x_noise_std",
            "title": "Reward for different x_noise_mean",
            "nominal_val": X_NOISE_STD,
            "use_log_scale": False
        },
    },
}


for env_name in ALL_ENV_NAMES:
    nominal_noises = ENV_NOMINAL_NOISES[env_name]
    configs = [BASE_CONFIG]
    configs += ENV_OUR_CONFIG[env_name]
    env_nominal_kwargs = ENV_NOMINAL_VALS[env_name]
    env_noise_keys = [key for key in env_nominal_kwargs.keys() if re.search("noise", key)]
    # Change env_nominal_noises
    if len(nominal_noises) != len(env_noise_keys):
        raise ValueError(f"Gave wrong number of noises expected:\t{' '.join(env_noise_keys)}")
    else:
        for idx, key in enumerate(env_noise_keys):
            env_nominal_kwargs[key] = nominal_noises[idx]
            print(f"Test env {key}={nominal_noises[idx]}")

    nominal_vals_text = "_".join([f"{k}_{env_nominal_kwargs[k]}" for k in env_noise_keys])

    save_path = f"figs/PPO/{env_name}"
    Path(save_path).mkdir(parents=True, exist_ok=True)

    env_data_fields = DATA_FIELDS[env_name]
    for field in fields:
        if field == "all_noise":
            field_txt = "noise"
        else:
            field_txt = field
        relevant_keys = [key for key in env_data_fields.keys() if field_txt in key.lower()]
        for data_key in relevant_keys:
            fig_1 = plt.figure(figsize=(5*SCALE, 4*SCALE), dpi=300)
            if env_data_fields[data_key]["nominal_val"] is not None:
                plt.axvline(x=env_data_fields[data_key]["nominal_val"], linestyle="--",
                            color=NOMINAL_COLOR, label='nominal value', linewidth=LINE_WIDTH)
            plt.xlabel(env_data_fields[data_key]["x_axis"], fontsize=30)
            if env_data_fields[data_key]["use_log_scale"]:
                plt.xscale("log", base=10)
            plt.ylabel("Reward (IQM)", fontsize=30)
            plt.tick_params(axis='both', which='major', labelsize=24)
            plt.title(env_name, fontdict={'size': 30})
            min_y = np.inf
            max_y = -np.inf
            min_x = np.inf
            max_x = -np.inf
            all_config_res = {}
            all_config_labels = {}
            all_config_colors = {}
            all_config_shapes = {}
            for config in configs:
                ALGO = config.get("ALGO", None)
                N_SAMPLE = config.get("N_SAMPLE", None)
                KAPPA = config.get("KAPPA", None)
                CVAR_ALPHA = config.get("cvar_alpha", None)

                algo_kwargs = {}
                algo_text = ""
                if ALGO == "Vanilla" or ALGO == "PPO":
                    algo_text = ALGO
                elif ALGO == "Ours" or ALGO == "EWoKPPO":
                    algo_kwargs["n_sample"] = N_SAMPLE
                    algo_kwargs["kappa"] = KAPPA
                    algo_text = f"{ALGO}_n_sample_{N_SAMPLE}_kappa_{KAPPA}"
                elif "Domain_Rand" in ALGO:
                    algo_text = ALGO + f"_{field}"
                else:
                    raise ValueError("unrecognized algorithm")

                log_dir = f"logs/{env_name}/{nominal_vals_text}/{algo_text}"

                seed_results = []
                key_vals = []
                all_dirs = []
                for root, dirs, files in os.walk(log_dir):
                    for subdir in dirs:
                        all_dirs.append(f"{root}/{subdir}")
                for subdir in all_dirs:
                    with open(f"{subdir}/{env_data_fields[data_key]['pkl_name']}.pkl", "rb") as fp:
                        results = pickle.load(fp)
                    if key_vals == []:
                        key_vals = [res[0] for res in results]
                    else:
                        if key_vals != [res[0] for res in results]:
                            raise ValueError(f"Not all seeds logged the same values for {data_key}")
                    avg_rewards = [res[1] for res in results]
                    seed_results.append(avg_rewards)

                all_config_res[f"{algo_text}"] = np.array(seed_results)
                all_config_labels[f"{algo_text}"] = config["label"]
                all_config_colors[f"{algo_text}"] = config["color"]
                all_config_shapes[f"{algo_text}"] = config["shape"]

            # Calculate RLiable metric
            aggregate_func = lambda x: np.array([metrics.aggregate_iqm(x)])

            final_res = {}
            for key in all_config_res.keys():
                final_res[key] = {"means":[], "cis": []}
            for i in range(len(key_vals)):
                aggregate_scores, aggregate_score_cis = rly.get_interval_estimates(
                    {key: all_config_res[key][:, i].reshape((all_config_res[key][:, i].size, 1)) for key in all_config_res.keys()},
                    aggregate_func, reps=10_000)
                for key in all_config_res.keys():
                    final_res[key]["means"].append(aggregate_scores[key][0])
                    final_res[key]["cis"].append(aggregate_score_cis[key])

            for key in all_config_res.keys():
                means = final_res[key]["means"]
                lower_bound = [bound[0].item() for bound in final_res[key]["cis"]]
                upper_bound = [bound[1].item() for bound in final_res[key]["cis"]]
                plt.plot(key_vals, means, f'{all_config_shapes[key]}-', color=all_config_colors[key], label=all_config_labels[key],
                         markersize=MARKER_SIZE, linewidth=LINE_WIDTH)
                plt.fill_between(key_vals, lower_bound, upper_bound, alpha=ALPHA_LEVEL, color=all_config_colors[key])
                min_y = min(min_y, min(lower_bound))
                max_y = max(max_y, max(upper_bound))
                min_x = min(min_x, min(key_vals))
                max_x = max(max_x, max(key_vals))

            y_diff = max_y - min_y
            x_diff = max_x - min_x
            plt.ylim(min_y - BORDER_STRECH_PERCENTAGE * y_diff, max_y + BORDER_STRECH_PERCENTAGE * y_diff)
            plt.xlim(min_x - BORDER_STRECH_PERCENTAGE * x_diff, max_x + BORDER_STRECH_PERCENTAGE * x_diff)
            ax = fig_1.axes[0]
            plt.locator_params(axis='both', nbins=4)
            ax.xaxis.set_tick_params(width=3)
            ax.yaxis.set_tick_params(width=3)
            plt.tight_layout()
            plt.legend(fontsize="24")
            # plt.show()
            plt.savefig(f"{save_path}/{data_key}")
            plt.close(fig_1)

exit(0)

