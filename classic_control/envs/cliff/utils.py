import copy

import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import random
import torch
import os
from envs.cliff.tmp_cliff import DELTAS
import matplotlib.patches as patches
from matplotlib.cbook import get_sample_data

EPSILON = 1e-5
ARROW_PATHS = [os.path.join(os.path.dirname(__file__), 'figs/arrow_up.png'),
               os.path.join(os.path.dirname(__file__), 'figs/arrow_right.png'),
               os.path.join(os.path.dirname(__file__), 'figs/arrow_down.png'),
               os.path.join(os.path.dirname(__file__), 'figs/arrow_left.png')]

def reshape_V_func(v_func, env):
    new_v = np.zeros(env.shape)
    for s in range(env.nS):
        try:
            pos = np.unravel_index(s, env.shape)
            new_v[pos] = v_func[s]
        except:
            print("Got new environment last state - Not printing it's value")
    return new_v

def print_policy(z, file_path=None, scale=None):
    z_copied = copy.deepcopy(z)
    z_copied[3][11] = 0.
    if scale is not None:
        z_copied[3][10] = scale[0]
        z_copied[3][11] = scale[1]
    nx, ny = z.shape
    indx, indy = np.arange(nx), np.arange(ny)
    x, y = np.meshgrid(indx, indy)

    fig, ax = plt.subplots(figsize=(12, 8))
    bg = np.zeros((nx, ny, 3))
    bg[z.shape[0] - 1, range(1, z.shape[1] - 1)] = [0.66, 0.66, 0.66]
    bg[z.shape[0] - 1, z.shape[1] - 1] = [119. / 255., 221. / 255., 119. / 255.]
    im = ax.imshow(bg, interpolation="nearest")
    alpha_values = np.ones((nx, ny))
    alpha_values[z.shape[0] - 1, range(1, z.shape[1])] = 0
    ax.imshow(z_copied, interpolation="nearest", cmap=cm.coolwarm, alpha=alpha_values)

    lines = []
    arrows = []
    for xval, yval in zip(x.flatten(), y.flatten()):
        if xval == z.shape[0] - 1 and yval in range(1, z.shape[1]):
            continue
        max_step = None
        max_step_idx = -1
        max_value = -np.inf
        for idx, d in enumerate(DELTAS):
            if xval + d[0] in range(0, nx) and yval+d[1] in range(0, ny):
                if z[xval + d[0], yval + d[1]] > max_value:
                    max_value = z[xval+d[0], yval+d[1]]
                    max_step = d
                    max_step_idx = idx
                if ((z[xval + d[0], yval + d[1]] * z[xval, yval] < 0 and not
                (xval == z.shape[0] - 1 and yval in range(1, z.shape[1]))) and not
                (xval + d[0] == z.shape[0] - 1 and yval + d[1] in range(1, z.shape[1]))):
                    if d[0] == 0:
                        line = [(yval + d[1] / 2.,xval - 0.5), (yval + d[1] / 2., xval + 0.5)]
                    else:
                        line = [(yval - 0.5, xval + d[0] / 2.), (yval + 0.5, xval + d[0] / 2.)]
                    lines.append(line)

        # Draw arrow
        # Read image file
        with get_sample_data(ARROW_PATHS[max_step_idx]) as file:
            arr_image = plt.imread(file, format='png')
        # Add axes to the right place of the arrow
        axin = ax.inset_axes([yval-0.25, xval-0.25, 0.5, 0.5],
                             transform=ax.transData)  # create new inset axes in data coordinates
        axin.imshow(arr_image)
        axin.axis('off')

    # for line in lines:
    #     path = patches.Polygon(line, facecolor='none', edgecolor=[1., 1., 0.1, 0.7],
    #                            linewidth=3, closed=False, joinstyle='round')
    #     ax.add_patch(path)

    # ax.text(z.shape[1] - 1, z.shape[0] - 1, "Goal", color='k', va='center', ha='center', size=20, weight='bold')
    # cliff_text = "Cliff!"
    # ax.text(z.shape[1] // 2, z.shape[0] - 1,cliff_text, color='w', va='center', ha='center', size=20, weight='bold')

    # Hide x and y ticks
    plt.xticks([], visible=False)
    plt.yticks([], visible=False)
    plt.colorbar(im, orientation="horizontal")
    plt.tight_layout()
    if file_path is None:
        plt.show()
    else:
        plt.savefig(file_path)
        plt.close()


def print_value(z, file_path=None, scale=None):
    z_copied = copy.deepcopy(z)
    z_copied[3][11] = 0.
    if scale is not None:
        z_copied[3][10] = scale[0]
        z_copied[3][11] = scale[1]
    nx, ny = z.shape
    indx, indy = np.arange(nx), np.arange(ny)
    x, y = np.meshgrid(indx, indy)
    fig, ax = plt.subplots(figsize=(12, 8))
    bg = np.zeros((nx, ny, 3))
    bg[z.shape[0] - 1, range(1, z.shape[1] - 1)] = [0.66, 0.66, 0.66]
    bg[z.shape[0] - 1, z.shape[1] - 1] = [119. / 255., 221. / 255., 119. / 255.]
    ax.imshow(bg, interpolation="nearest")
    alpha_values = np.ones((nx, ny))
    alpha_values[z.shape[0] - 1, range(1, z.shape[1])] = 0
    im = ax.imshow(z_copied, interpolation="nearest", cmap=cm.coolwarm, alpha=alpha_values)

    # Flatten the 2D array to a 1D array
    flat_data = np.array(z_copied).flatten()

    # Calculate the 0.75 quantile
    quantile_75 = 0.75 * (np.max(flat_data) - np.min(flat_data))

    for xval, yval in zip(x.flatten(), y.flatten()):
        if xval == z.shape[0] - 1 and yval in range(1, z.shape[1]):
            continue
        zval = z[xval, yval]
        t = "{:.2f}".format(zval)
        c = 'w' if zval > quantile_75 else 'k'
        ax.text(yval, xval, t, color=c, va='center', ha='center', size=18)

    # ax.text(z.shape[1] - 1, z.shape[0] - 1, "Goal", color='k', va='center', ha='center', size=18)
    # cliff_text = "Cliff!"
    # for idx, y in enumerate(range(3, 9)):
    #     ax.text(y, z.shape[0] - 1, cliff_text[idx], color='w', va='center', ha='center', size=20)

    # Hide x and y ticks
    plt.xticks([], visible=False)
    plt.yticks([], visible=False)
    plt.colorbar(im, orientation="horizontal")
    plt.tight_layout()
    if file_path is None:
        plt.show()
    else:
        plt.savefig(file_path)
        plt.close()


def get_offset(idx):
    DELTAS = [[-1, 0], [0, 1], [1, 0], [0, -1]]
    return np.array(DELTAS[idx])


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def print_P(env, file_path=None):
    nx, ny = env.shape
    indx, indy = np.arange(nx), np.arange(ny)
    x, y = np.meshgrid(indx, indy)

    fig, ax = plt.subplots(figsize=(51, 17))
    z = np.zeros((nx * 9, ny * 9))
    for xval, yval in zip(x.flatten(), y.flatten()):
        pos = np.ravel_multi_index((xval, yval), env.shape)
        state_middle_idx = np.array([xval * 9 + 4, yval * 9 + 4])
        for action in range(env.nA):
            probs = [t[0] for t in env.P[pos][action]]
            action_middle_idx = state_middle_idx + 3 * get_offset(action)
            for idx, prob in enumerate(probs):
                # t = "{:.4f}".format(prob)
                offset = get_offset(idx)
                current_idx = action_middle_idx + offset
                z[current_idx[0], current_idx[1]] = prob
                # ax.text(current_idx[1], current_idx[0], t, color='k', va='center', ha='center')

    ax.imshow(z, interpolation="nearest", cmap=cm.YlGn)  # plot grid values

    # Flatten the 2D array to a 1D array
    flat_data = np.array(z).flatten()

    # Calculate the 0.75 quantile
    quantile_75 = 0.75 * (np.max(flat_data) - np.min(flat_data))

    indx, indy = np.arange(nx * 9), np.arange(ny * 9)
    x, y = np.meshgrid(indx, indy)
    for xval, yval in zip(x.flatten(), y.flatten()):
        zval = z[xval, yval]
        if zval < EPSILON:
            continue
        t = "{:.2f}".format(zval)
        c = 'w' if zval > quantile_75 else 'k'  # if dark-green, change text color to white
        ax.text(yval, xval, t, color=c, va='center', ha='center')

    for i in range(ny):
        ax.axvline(x=i * 9 - 0.5, color='black', linestyle='-', linewidth=1.5)
        ax.axvline(x=i * 9 + 3 - 0.5, color='grey', linestyle='--', linewidth=0.5)
        ax.axvline(x=(i + 1) * 9 - 3 - 0.5, color='grey', linestyle='--', linewidth=0.5)
    for i in range(nx):
        ax.axhline(y=i * 9 - 0.5, color='black', linestyle='-', linewidth=1.5)
        ax.axhline(y=i * 9 + 3 - 0.5, color='grey', linestyle='--', linewidth=0.5)
        ax.axhline(y=(i + 1) * 9 - 3 - 0.5, color='grey', linestyle='--', linewidth=0.5)

    # Hide x and y ticks
    plt.xticks([], visible=False)
    plt.yticks([], visible=False)
    plt.tight_layout()
    if file_path is None:
        plt.show()
    else:
        plt.savefig(file_path)
        plt.close()
