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
    hk = 1.0 # TODO: needed for HW5 but not HW4 ?
    prod = 1.0
    for j in range(len(x)):
        prod *= np.cos(k[j] * np.pi / (ub - lb) * (x[j] - lb))
    return prod / hk


def get_Phik(k, lb, ub):
    # integral over X of Phi(x)fk(x)dx
    integrand = lambda x1, x2: get_Fk(np.array([x1, x2]), k, lb, ub) * Phi(np.array([x1, x2]))
    return dblquad(integrand, lb, ub, lb, ub)[0]


def get_ck(x, k, T, dt, i, lb, ub): 
    intg = 0 # integral from (0,T) of fk(x(t))
    for j in range(i):
        intg += dt * get_Fk(x[j], k, lb, ub)
    return 1/T * intg


def metric(dt, x, lb, ub, K, Phik_list, i):
    T = (i+1) * dt
    possib_u = np.array([[0.0,0.0],[1.0,0.0],[0.0,1.0],[-1.0,0.0],[0.0,-1.0]])

    # Maximally Ergodic Trajectory Step: 
    # Simulate next step for various choices of u, choose u with lowest ergodicity
    eps_list = []
    for u in possib_u:
        x[i] = x[i-1] + dt * u
        s = ((i + 1) + 1.0)/2.0
        j = 0
        eps = 0
        for k1 in range(K):
            for k2 in range(K):
                k = np.array([k1, k2])
                Phik = Phik_list[j]

                ck = get_ck(x, k, T, dt, i, lb, ub)

                Lambda = (1 + np.linalg.norm(k)**2)**(-s)

                eps += abs(ck - Phik)**2 * Lambda
                j += 1
        eps_list.append(eps)

    ergodic_u = possib_u[np.argmin(eps_list)]
    print("t= " + str(round(T,1)) +  ",\tu_erg= " + str(ergodic_u) + ",\teps= " + str(eps_list))

    return ergodic_u


def main():
    
    T = 10
    dt = 0.1
    n = int(T/dt) + 1

    # Trajectory
    x0 = np.array([0,1])
    x = np.zeros((n,2))
    x[0] = x0

    # Range X
    lb, ub = -10, 10

    K = 10
    print("Computing Phik for all possible k ...")
    Phik_list = []
    for k1 in range(K):
            for k2 in range(K):
                k = np.array([k1, k2])
                Phik_list.append(get_Phik(k, lb, ub))

    # Ergodic Trajectory
    # TODO: should u be  bounded?
    # TODO: barrier funtion needed ? e.g. adapted logistic function
    for i in range(1, n):
        ergodic_u = metric(dt, x, lb, ub, K, Phik_list, i)
        x[i] = x[i-1] + dt * ergodic_u

    fig, ax = plt.subplots()
    ax.plot(x[:,0], x[:,1], color='slateblue')
    ax.set_xlabel(r'$x_{1}$')
    ax.set_ylabel(r'$x_{2}$')
    ax.set_title(r'Maximally ergodic trajectory, $dt=$ {}, $K=$ {}'.format(T, dt, K))
    fig.tight_layout()
    plt.savefig('./ME455_ActiveLearning/HW5/HW5_1_ergodic_traj.png', dpi=300)
    plt.show()


if __name__ == "__main__":
    main()