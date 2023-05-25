import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as colors

import HW4_1_b as b

# Plots LaTeX-Style
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'


def main():
    b_size = 100 +1
    b_0, b_n = 0, 0.5
    b_array = np.linspace(b_0, b_n, b_size)
    
    T_array = [1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
    dt = 0.1

    
    # Range X
    lb, ub = -10, 10

    K = 10
    print("Computing Phik for all possible k ...")
    Phik_list = []
    for k1 in range(K):
            for k2 in range(K):
                k = np.array([k1, k2])
                Phik_list.append(b.get_Phik(k, lb, ub))

    Z = np.zeros((len(b_array), len(T_array)), dtype=np.float64)

    for idx, T in enumerate(T_array):
        print("Computing epsilon for T = {}".format(T))
        n = int(T/dt) + 1

        epsilon = b.metric(b_array, T, dt, n, lb, ub, K, Phik_list)
        Z[:,idx] = epsilon

    X, Y = np.meshgrid(T_array, b_array)
    min_index  = np.unravel_index(Z.argmin(), Z.shape)
    print(" Optimal T={}, b={}".format(T_array[min_index[1]], b_array[min_index[0]]))

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_surface(X, Y, Z, cmap=cm.viridis, norm=colors.LogNorm(vmin=Z.min(), vmax=Z.max()), edgecolor='none', linewidth=0, alpha=0.8)
    ax.scatter(T_array[min_index[1]], b_array[min_index[0]], Z[min_index], 'o', color='crimson', label=r'$argmin(\epsilon)$')
    ax.legend()
    ax.set_xlabel(r'$T$')
    ax.set_ylabel(r'$b$')
    ax.zaxis.set_rotate_label(False)
    ax.set_zlabel(r'$\epsilon$', rotation=90)
    ax.set_title(r'Ergodic metric $\epsilon$ vs. $b$ vs. $T$, $dt=$ {}, $K=$ {}'.format(dt, K))
    ax.view_init(elev=20, azim=10, roll=0)
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    fig.tight_layout()
    plt.savefig('./ME455_ActiveLearning/HW4/Problem 1/HW4_1_ergodic_metric_bT.png', dpi=300)
    plt.show()


if __name__ == "__main__":
    main()