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
            sum += Lambda * (get_ck(x, k) - Phik_list[k1, k2]) **2
    intg_u = 0
    for i in range(n):
        intg_u += dt * (u[i].T @ R @ u[i])
    return q * sum + intg_u

def DJ(x, u, zeta):
    intg = 0
    for i in range(n):
        intg += dt * ((a(x, i)).T @ zeta[i,:2] + (b(u[i])).T @ zeta[i,2:])
    return intg

### Descent Direction ###
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
        r_dot = -(A - B @ np.linalg.inv(R) @ B.T @ P[i]).T @ r[i] - a(x, i) + P[i] @ B @ np.linalg.inv(R) @ b(u[i])
        P[i-1] = P[i] - dt*P_dot
        r[i-1] = r[i] - dt*r_dot
        
        i -= 1

    for i in range(n-1):
        zeta[i, 2:] = np.linalg.inv(R) @ B.T @ P[i] @ zeta[i, :2] - np.linalg.inv(R) @ B.T @ r[i] - np.linalg.inv(R) @ b(u[i])
        zeta[i+1, :2] = zeta[i, :2] + dt*(A @ zeta[i, :2] + B @ zeta[i, 2:])
    return zeta


def a(x, i):
    sum = np.zeros(2)
    for k1 in range(K):
        for k2 in range(K):
            k = np.array([k1, k2])
            Lambda = (1 + np.linalg.norm(k)**2)**(-s)
            sum += 2/T * Lambda * (get_ck(x, k) - Phik_list[k1, k2]) * get_DFk(x[i], k)
    return (q * sum).T


def b(ui):
    return (ui.T @ R).T


### Ergodicity ###
def get_hk(k):
    integrand = lambda x1, x2: (np.cos(k[0]*np.pi/(ub-lb) * (x1 - ub)) * np.cos(k[1]*np.pi/(ub-lb) * (x2 - ub)))**2
    return np.sqrt(dblquad(integrand, lb, ub, lb, ub)[0])
    # return 1.0


def get_Fk(x, k):
    hk = hk_list[k[0], k[1]]
    prod = 1.0
    for i in range(len(x)):
        prod *= np.cos(k[i] * np.pi / (ub - lb) * (x[i] - ub))
    return prod / hk


def get_DFk(xi, k):
    hk = hk_list[k[0], k[1]]
    L = ub - lb
    D1Fk = -k[0]*np.pi/(L*hk) * np.sin(k[0]*np.pi/L * (xi[0] - ub)) * np.cos(k[1]*np.pi/L * (xi[1] - ub))
    D2Fk = -k[1]*np.pi/(L*hk) * np.cos(k[0]*np.pi/L * (xi[0] - ub)) * np.sin(k[1]*np.pi/L * (xi[1] - ub))
    return np.array([D1Fk, D2Fk])

def get_ck(x, k):
    intg_F = 0
    for i in range(n):
        intg_F += dt * get_Fk(x[i], k)
    return intg_F/T


def get_Phik(k):
    # integral over X of Phi(x)fk(x)dx
    integrand = lambda x1, x2: get_Fk(np.array([x1, x2]), k) * Phi(np.array([x1, x2]))
    return dblquad(integrand, lb, ub, lb, ub)[0]

def Phi(x):
    return np.linalg.det(2*np.pi*Sig)**(-0.5) * np.exp(-1/2 * (x-mu).T @ np.linalg.inv(Sig) @ (x-mu))

### Main ###
# Global Parameters
T = 10
dt = 0.1
n = int(T/dt) + 1

mu = np.array([0, 0])
Sig = np.diag([2, 2])

# Range X
lb, ub = -5, 5

K = 10
s = 1.5

q = 10.
Q = np.array([[0.1, 0.],[0., 0.1]])
R = np.array([[1., 0.],[0., 1.]])
P_1 = np.array([[0., 0.],[0., 0.]])

# Trajectory: Initial trajectory
x0 = np.array([0,1])
u_erg = np.zeros((n,2))
for i in range(n):
    # u_erg[i] = [0.2*np.sin(np.pi/100 *i), 0.2*np.cos(np.pi/100 *i)]
    u_erg[i] = [-0.5*np.cos(np.pi/50 *i), -0.62*np.sin(np.pi/50 *i)]
    # u_erg[i] = [0.1, -0.1]
x_init = dynamics(x0, u_erg)
x_erg = x_init

print("Computing all possible h_k, Phi_k ...")
# Get Phi_k
Phik_list = np.zeros((K,K))
hk_list = np.zeros((K,K))
for k1 in range(K):
    for k2 in range(K):
        hk_list[k1,k2] = get_hk(np.array([k1, k2]))
        Phik_list[k1,k2] = get_Phik(np.array([k1, k2]))


# Ergodic Trajectory
print("Computing maximally ergodic trajectory ...")
# TODO: barrier funtion needed ? e.g. adapted logistic function
# Ergodicity Threshold
eps = 1E-3

zeta = np.zeros((n,4)) # zeta = [z0, z1, v0, v1]
J_array = [float('inf')]

i = 0
dj = float('inf')

fig2, ax = plt.subplots()
ax.plot(x_init[:,0], x_init[:,1], color='darkorange')
ax.set_xlabel(r'$x_{1}$')
ax.set_ylabel(r'$x_{2}$')
ax.set_title(r'Maximally ergodic trajectory, $dt=$ {}, $K=$ {}'.format(dt, K))
fig2.tight_layout()

n_iter = 10
while i==0 or dj > eps and i < n_iter:
    zeta = get_zeta(x_erg, u_erg)
    print("i= {},\tDJ= {},\tJ= {}".format(i, np.round(dj, 4), np.round(J_array[-1], 4)))
    
    # Armijo
    J_u = J(x_erg, u_erg)
    J_array.append(J_u)
    DJ_u_zeta = DJ(x_erg, u_erg, zeta)
    gamma = 0.1
    # alpha = 0.1
    # beta = 0.5
    # while J(dynamics(x0, u_erg + gamma*zeta[:, 2:]), u_erg + gamma*zeta[:, 2:]) > J_u + alpha*gamma*DJ_u_zeta:
    #     gamma *= beta
    #     # if gamma < 1e-16:
    #     #     gamma = 1e-16
    #     #     break
    # print(gamma)

    # Update u, x for all times
    for j in range(n):
        u_erg[j] += gamma*zeta[j, 2:]
    x_erg = dynamics(x0, u_erg)

    dj = np.abs(DJ_u_zeta)

    ax.plot(x_erg[:,0], x_erg[:,1], color='slateblue', alpha=(1 - i/(n_iter+1)))
    fig2.canvas.draw()
    plt.pause(0.01)

    i += 1

### Plotting ###
fig, ax = plt.subplots()
ax.plot(x_init[:,0], x_init[:,1], color='darkorange', label=r'$x_{0}(t)$')
ax.plot(x_erg[:,0], x_erg[:,1], color='slateblue', label=r'$x_{erg}(t)$')
ax.legend()
ax.set_xlabel(r'$x_{1}$')
ax.set_ylabel(r'$x_{2}$')
ax.set_title(r'Maximally ergodic trajectory, $dt=$ {}, $K=$ {}'.format(dt, K))
fig.tight_layout()
plt.savefig('./ME455_ActiveLearning/HW5/HW5_1_ergodic_traj.png', dpi=300)

fig2, ax = plt.subplots()
ax.step(np.linspace(1, len(J_array[1:]), len(J_array[1:])), J_array[1:], color='darkorange')
ax.set_xlabel(r'$i$')
ax.set_ylabel(r'$J$')
ax.set_title(r'Cost vs. Iteration')
fig.tight_layout()
plt.savefig('./ME455_ActiveLearning/HW5/HW5_1_cost.png', dpi=300)
plt.show()