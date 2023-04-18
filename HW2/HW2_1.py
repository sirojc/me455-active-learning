import numpy as np
import matplotlib.pyplot as plt

T = 2 * np.pi
n = 1000
h = T / n

Q = [[10, 0, 0],[0, 10, 0],[0, 0, 1]]
R = [[1, 0],[0, 1]]

def dynamics(x0, u):
    x = np.zeros((n,3))
    x[0] = x0
    for i in range(n-1):
        x[i+1,2] = x[i,2] + h*u[i,1]
        x[i+1,0] = x[i,0] + h*np.cos(x[i,2])*u[i,0]
        x[i+1,1] = x[i,1] + h*np.sin(x[i,2])*u[i,0]
    return x

# Cost Function
def J(x, u):
    intg = 0
    for i in range(len(x)):
        intg += h*(np.dot(np.dot((x[i] - x_g[i]).T, Q), (x[i]-x_g[i])) + np.dot(np.dot(u[i].T, R), u[i]))
    return intg

# Cost Function Directional Derivative
def DJ(x, u, zheta):
    intg = 0
    for i in range(len(x)):
        intg += h*2*(np.dot(np.dot((x[i] - x_g[i]).T, Q), zheta[i, :3]) + np.dot(np.dot(u[i].T, R), zheta[i, 3:]))
    return intg


# Goal Trajectory
x_g = np.zeros((n,3))
for i in range(n):
    x_g[i,0] = 2*h*i/np.pi
    x_g[i,2] = np.pi/2

# Initial Trajectory
x0 = [0,0,np.pi/2]
u_init = np.zeros((n,2))
for i in range(n-1):
    u_init[i] = [1, -0.5]
x_init = dynamics(x0, u_init)

# Initial Cost
J_init = J(x_init, u_init)
print(J_init)



# iLQR
eps = 0.1 # Threshold

u_iLQR = u_init
x_iLQR = x_init
zheta = np.zeros((n,5)) # zheta = [z0, z1, z2, v0, v1]

i = 0
while i==0 or np.abs(DJ(x_iLQR, u_iLQR, zheta)) > eps:
    # Get zheta (perturbation)
    # ???

    # Armijo
    alpha = 0.5
    beta = 0.1
    gamma = 1
    J_u = J(x_iLQR, u_iLQR)
    DJ_u_zheta = DJ(x_iLQR, u_iLQR, zheta)
    while J(dynamics(x0, u_iLQR + gamma*zheta[:, :3]), u_iLQR + gamma*zheta[:, 3:]) > J_u + alpha*gamma*DJ_u_zheta: # Goal Trajectory may not have same amount of steps as step size variable?
        gamma *= beta

    # Update x, u for all times
    # ???
    i += 1



# Plots: Initial Trajectory, Final Optimized Trajectory, Optimized Control Signal



