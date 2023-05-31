import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import dblquad

# Plots LaTeX-Style
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'

### Dynamics ###
def dynamics(x0, u_erg):
    x = np.zeros((n,2))
    x[0] = x0
    for i in range(n-1):
        x[i+1] = x[i] + dt * u_erg[i]
    return x


### Cost Function ###
def J(x, u):
    sum = 0
    for k1 in range(K):
        for k2 in range(K):
            k = np.array([k1, k2])
            Lambda = (1 + np.linalg.norm(k)**2)**(-s)
            intg_F = 0
            for i in range(n):
                intg_F += dt * get_Fk(x[i], k)
            sum += Lambda * (intg_F/T - Phik_list[k1, k2]) **2
    intg_u = 0
    for i in range(n):
        intg_u += dt * (u[i].T @ R @ u[i])
    return q * sum + intg_u

def DJ(x, u, zeta):
    intg = 0
    for i in range(n):
        intg += dt * ((a(x, i)).T @ zeta[i,:2] + (b(u[i])).T @ zeta[i,2:])
    return intg


def a(x, i):
    sum = 0
    for k1 in range(K):
        for k2 in range(K):
            k = np.array([k1, k2])
            Lambda = (1 + np.linalg.norm(k)**2)**(-s)
            intg_F = 0
            intg_DF = 0
            for j in range(n):
                intg_F += dt * get_Fk(x[j], k)
            sum += 2 * Lambda * (intg_F/T - Phik_list[k1, k2]) * get_DFk(x[i], k)/T
    return (q * sum).T

def b(ui):
    return (ui.T @ R).T

def get_zeta(x, u):
    A = np.array([[0, 0],[0, 0]]) # D1 f(x,u)
    B = np.array([[1, 0],[0, 1]]) # D2 f(x,u)

    zeta = np.zeros((n,4))
    P = np.zeros((n,2,2)) # Time, Matrix
    r = np.zeros((n,2)) # Time, Vector
    P[n-1] = P_1
    r[n-1] = [0, 0]

    i = n-1
    while i > 0:
        P_dot = -(P[i] @ A) + A.T @ P[i] + P[i] @ B @ np.linalg.inv(R) @ B.T @ P[i] - Q
        # print("P_dot=", P_dot)
        r_dot = -(A - B @ np.linalg.inv(R) @ B.T @ P[i]).T @ r[i] - a(x, i) + P[i] @ B @ np.linalg.inv(R) @ b(u[i])
        # print("r_dot=", r_dot)
        P[i-1] = P[i] - dt*P_dot
        r[i-1] = r[i] - dt*r_dot
        
        i -= 1

    zeta[0, :2] = [0, 0]
    for i in range(n-1):
        zeta[i, 2:] = np.linalg.inv(R) @ B.T @ P[i] @ zeta[i, :2] - np.linalg.inv(R) @ B.T @ r[i] - np.linalg.inv(R) @ b(u[i])
        zeta[i+1, :2] = zeta[i, :2] + dt*(A @ zeta[i, :2] + B @ zeta[i, 2:]) # A = zeros()
    return zeta


### Ergodicity ###
def get_Fk(x, k):
    hk = hk_list[k[0], k[1]]
    prod = 1.0
    for j in range(len(x)):
        prod *= np.cos(k[j] * np.pi / (ub - lb) * (x[j] - lb))
    return prod / hk

def get_hk(k):
    integrand = lambda x1, x2: (np.cos(k[0]*np.pi/(ub-lb) * (x1 - lb)) * np.cos(k[1]*np.pi/(ub-lb) * (x2 - lb)))**2
    return (dblquad(integrand, lb, ub, lb, ub)[0])**0.5

def get_DFk(xi, k):
    hk = hk_list[k[0], k[1]]
    L = ub - lb
    D1Fk = -k[0]*np.pi/(L*hk) * np.sin(k[0]*np.pi/L * (xi[0] - lb)) * np.cos(k[1]*np.pi/L * (xi[1] - lb))
    D2Fk = -k[1]*np.pi/(L*hk) * np.cos(k[0]*np.pi/L * (xi[0] - lb)) * np.sin(k[1]*np.pi/L * (xi[1] - lb))
    return np.array([D1Fk, D2Fk])

def get_Phik(k):
    # integral over X of Phi(x)fk(x)dx
    integrand = lambda x1, x2: get_Fk(np.array([x1, x2]), k) * Phi(np.array([x1, x2]))
    return dblquad(integrand, lb, ub, lb, ub)[0]

def Phi(x):
    return np.linalg.det(2*np.pi*Sig)**(-1/2) * np.exp(-1/2 * np.dot(np.dot((x-mu).T, np.linalg.inv(Sig)), (x-mu)))


### Main ###
def main():

    # Trajectory: Initial trajectory no movement
    x0 = np.array([0,1])
    x_erg = np.zeros((n,2))
    u_erg = np.zeros((n,2))
    for i in range(n):
        x_erg[i] = x0
    
    # Ergodic Trajectory
    print("Computing maximally ergodic trajectory ...")
    # TODO: barrier funtion needed ? e.g. adapted logistic function
    # Ergodicity Threshold
    eps = 1E-3

    zeta = np.zeros((n,4)) # zeta = [z0, z1, v0, v1]
    J_array = []

    i = 0
    dj = float('inf')
    while i==0 or dj > eps:
        zeta = get_zeta(x_erg, u_erg)
        print("DJ=", dj, "\tzeta[-1]=", zeta[-1])
        
        # Armijo
        alpha = 0.1
        beta = 0.5
        gamma = 1
        J_u = J(x_erg, u_erg)
        J_array.append(J_u)
        DJ_u_zeta = DJ(x_erg, u_erg, zeta)
        while J(dynamics(x0, u_erg + gamma*zeta[:, 2:]), u_erg + gamma*zeta[:, 2:]) > J_u + alpha*gamma*DJ_u_zeta:
            gamma *= beta

        # Update u, x for all times
        for j in range(n):
            u_erg[j] += gamma*zeta[j, 2:]
        x_erg = dynamics(x0, u_erg)

        dj = np.abs(DJ_u_zeta)
        print(dj)
        i += 1

    ### Plotting ###
    fig, ax = plt.subplots()
    ax.plot(x_erg[:,0], x_erg[:,1], color='slateblue')
    ax.set_xlabel(r'$x_{1}$')
    ax.set_ylabel(r'$x_{2}$')
    # ax.set_xlim([-2, 2])
    # ax.set_ylim([-2, 2])
    ax.set_title(r'Maximally ergodic trajectory, $dt=$ {}, $K=$ {}'.format(dt, K))
    fig.tight_layout()
    plt.savefig('./ME455_ActiveLearning/HW5/HW5_1_ergodic_traj.png', dpi=300)
    plt.show()


### Global Parameters ###
T = 10
dt = 0.1
n = int(T/dt) + 1

mu = np.array([0, 0])
Sig = np.diag([2, 2])

# Range X
lb, ub = -5, 5

K = 10
s = (n+1)/2.0

q = 10 # np.array([[1, 0],[0,1]])
Q = np.array([[q, 0],[0, q]])
R = np.array([[1,0],[0,1]])
P_1 = np.array([[0, 0],[0, 0]])

print("Computing all possible h_k, Phi_k ...")
# Get Phi_k
Phik_list = np.zeros((K,K))
hk_list = np.zeros((K,K))
for k1 in range(K):
    for k2 in range(K):
        hk_list[k1,k2] = get_hk(np.array([k1, k2]))
        Phik_list[k1,k2] = get_Phik(np.array([k1, k2]))


if __name__ == "__main__":
    main()