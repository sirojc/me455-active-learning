import matplotlib.pyplot as plt
import numpy as np

# Plots LaTeX-Style
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'

def plot_trajectory(gg, door, loc, s):
    fig, axs = plt.subplots()
    axs.set_xlim([0, gg])
    axs.set_ylim([0, gg])

    major_ticks=np.linspace(0,gg,6)
    minor_ticks=np.linspace(0,gg,gg+1)
    axs.set_xticks(major_ticks)
    axs.set_yticks(major_ticks)
    axs.set_xticks(minor_ticks, minor=True)
    axs.set_yticks(minor_ticks, minor=True)
    axs.grid(which='both', alpha=0.8)

    axs.plot(loc[:, 0], loc[:, 1], color='k', linewidth='2', label='Trajectory')
    axs.plot(loc[0, 0], loc[0, 1], color='r', marker='s', markersize=10, label='Start')
    axs.plot(door[0], door[1], color='g', marker='s', markersize=10, label='Door')

    axs.legend()

    axs.set_title('Run ' + str(s+1))
    plt.tight_layout()
    plt.savefig('./ME455_ActiveLearning/HW4/Problem 2/plots/trajectory' + str(s+1) + '.png')
    plt.show()