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
    hk = 1 # minimum at same location if no normalization

    prod = 1
    for i in range(len(k)):
        prod *= np.cos(k[i] * np.pi / (ub[i] - lb[i]) * (x[i] - lb[i]))
    return prod / hk


def get_Phi_k(Fk, lb, ub):
    # integral over X of Phi(x)fk(x)dx
    intg = dblquad(lambda x1, x2: Fk * Phi(np.array([x1, x2])), lb, ub, lb, ub)
    return intg


def get_Ck(x, k, T, dt, n, lb, ub): 
    intg = 0 # integral from (0,T) of fk(x(t))
    for i in range(n):
        intg += dt * get_Fk(x[i], k, lb, ub)
    return 1/T * intg


def main():
    b_size = 10
    b_0, b_n = 0, b_size
    b_array = np.linspace(b_0, b_n, b_size)
    metric = np.array(b_size)
    
    T = 100
    dt = 0.1
    n = int(T/dt) + 1
    
    x0 = np.array([0, 1])

    # Range X
    lb, ub = -200, 200

    for b in b_array:
        A = np.array([[0, 1],[-1, -b]])
        
        # Trajectory
        x = np.zeros((n, 2))
        x[0] = x0
        for t in range(1, n):
            x[t] = x[t-1] + dt * np.dot(A, x[t-1])

        plt.plot(x[:,0], x[:,1])
        plt.show()

        # Ergodic Metric



if __name__ == "__main__":
    main()