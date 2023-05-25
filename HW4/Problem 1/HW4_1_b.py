import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import dblquad

# Plots LaTeX-Style
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'

def Phi(x):
    mu = np.array([0, 0])
    Sig = np.diag([2, 2])
    return np.linalg.det(2*np.pi*Sig)**(-1/2) * np.exp(-1/2 * np.dot(np.dot((x-mu).T, np.linalg.inv(Sig)), (x-mu)))


def get_Fk(x, k, lb, ub):
    hk = 1.0 # minimum at same location if no normalization
    prod = 1.0
    for i in range(len(x)):
        prod *= np.cos(k[i] * np.pi / (ub - lb) * (x[i] - lb))
    return prod / hk


def get_Phik(k, lb, ub):
    # integral over X of Phi(x)fk(x)dx
    integrand = lambda x1, x2: get_Fk(np.array([x1, x2]), k, lb, ub) * Phi(np.array([x1, x2]))
    return dblquad(integrand, lb, ub, lb, ub)[0]


def get_ck(x, k, T, dt, n, lb, ub): 
    intg = 0 # integral from (0,T) of fk(x(t))
    for i in range(n):
        intg += dt * get_Fk(x[i], k, lb, ub)
    return 1/T * intg


def metric(b_array, T, dt, n, lb, ub, K, Phik_list):
    epsilon = np.zeros(len(b_array), dtype=np.float64)

    for idx, b in enumerate(b_array):
        A = np.array([[0, 1],[-1, -b]])
        
        # Trajectory
        x0 = np.array([0, 1])
        x = np.zeros((n, 2))
        x[0] = x0
        for t in range(1, n):
            x[t] = x[t-1] + dt * np.dot(A, x[t-1])

        s = (x0.shape[0]+1)/2.0

        # Ergodic Metric
        eps = 0.0
        i = 0
        for k1 in range(K):
            for k2 in range(K):
                k = np.array([k1, k2])
                # Phik = get_Phik(k, lb, ub)
                Phik = Phik_list[i]

                ck = get_ck(x, k, T, dt, n, lb, ub)

                Lambda = (1 + np.linalg.norm(k)**2)**(-s)

                eps += abs(ck - Phik)**2 * Lambda
                i += 1

        epsilon[idx] = eps
        print("Ergodic metric for b= {} is eps= {}".format(b, eps))
    return epsilon


def main():
    b_size = 100 +1
    b_0, b_n = 0, 0.5
    b_array = np.linspace(b_0, b_n, b_size)
    
    T = 100
    dt = 0.1
    n = int(T/dt) + 1

    # Range X
    lb, ub = -10, 10

    K = 10
    print("Computing Phik for all possible k ...")
    Phik_list = []
    for k1 in range(K):
            for k2 in range(K):
                k = np.array([k1, k2])
                Phik_list.append(get_Phik(k, lb, ub))

    epsilon = metric(b_array, T, dt, n, lb, ub, K, Phik_list)

    min_index = np.argmin(epsilon)
    print("Minimum ergodic metric is at b= {}".format(b_array[min_index]))

    fig, ax = plt.subplots()
    ax.plot(b_array, epsilon, color='slateblue')
    ax.plot(b_array[min_index], epsilon[min_index], 'o', color='crimson', label=r'$argmin(\epsilon)$')
    ax.legend()
    ax.set_xlabel(r'$b$')
    ax.set_ylabel(r'$\epsilon$')
    ax.set_title(r'Ergodic metric $\epsilon$ vs. $b$, $T=$ {}, $dt=$ {}, $K=$ {}'.format(T, dt, K))
    fig.tight_layout()
    plt.savefig('./ME455_ActiveLearning/HW4/Problem 1/HW4_1_ergodic_metric_b.png', dpi=300)
    plt.show()


if __name__ == "__main__":
    main()