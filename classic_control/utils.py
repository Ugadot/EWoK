import matplotlib.pyplot as plt
import numpy as np


def get_colors(num_bins=10):
    '''
    Create colors from light to dark in different colors
    '''
    reds = []
    rgbs = (plt.cm.Reds(np.linspace(0, 1, num_bins))[:, :3] * 255).astype(int)
    for i in range(rgbs.shape[0]):
        rgb = rgbs[i, :]
        hex = '#%02x%02x%02x' % tuple(rgb)
        # print(hex)
        reds.append(hex)

    blues = []
    rgbs = (plt.cm.Blues(np.linspace(0, 1, num_bins))[:, :3] * 255).astype(int)
    for i in range(rgbs.shape[0]):
        rgb = rgbs[i, :]
        hex = '#%02x%02x%02x' % tuple(rgb)
        # print(hex)
        blues.append(hex)

    greens = []
    rgbs = (plt.cm.Greens(np.linspace(0, 1, num_bins))[:, :3] * 255).astype(int)
    for i in range(rgbs.shape[0]):
        rgb = rgbs[i, :]
        hex = '#%02x%02x%02x' % tuple(rgb)
        # print(hex)
        greens.append(hex)

    oranges = []
    rgbs = (plt.cm.Oranges(np.linspace(0, 1, num_bins))[:, :3] * 255).astype(int)
    for i in range(rgbs.shape[0]):
        rgb = rgbs[i, :]
        hex = '#%02x%02x%02x' % tuple(rgb)
        # print(hex)
        oranges.append(hex)

    purples = []
    rgbs = (plt.cm.Purples(np.linspace(0, 1, num_bins))[:, :3] * 255).astype(int)
    for i in range(rgbs.shape[0]):
        rgb = rgbs[i, :]
        hex = '#%02x%02x%02x' % tuple(rgb)
        # print(hex)
        purples.append(hex)

    bones = []
    rgbs = (plt.cm.bone(np.linspace(0, 1, num_bins))[:, :3] * 255).astype(int)
    for i in range(rgbs.shape[0]):
        rgb = rgbs[i, :]
        hex = '#%02x%02x%02x' % tuple(rgb)
        # print(hex)
        bones.append(hex)

    greys = []
    rgbs = (plt.cm.Greys(np.linspace(0, 1, num_bins))[:, :3] * 255).astype(int)
    for i in range(rgbs.shape[0]):
        rgb = rgbs[i, :]
        hex = '#%02x%02x%02x' % tuple(rgb)
        # print(hex)
        greys.append(hex)

    infernos = []
    rgbs = (plt.cm.inferno(np.linspace(0, 1, num_bins))[:, :3] * 255).astype(int)
    for i in range(rgbs.shape[0]):
        rgb = rgbs[i, :]
        hex = '#%02x%02x%02x' % tuple(rgb)
        # print(hex)
        infernos.append(hex)

    wistias = []
    rgbs = (plt.cm.Wistia(np.linspace(0, 1, num_bins))[:, :3] * 255).astype(int)
    for i in range(rgbs.shape[0]):
        rgb = rgbs[i, :]
        hex = '#%02x%02x%02x' % tuple(rgb)
        # print(hex)
        wistias.append(hex)

    return reds, blues, greens, oranges, purples, bones, greys, infernos, wistias
