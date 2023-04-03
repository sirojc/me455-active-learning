import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg, integrate


Q = np.array([[2, 0],[0, 0.01]])
R = 0.1
P = np.array([[1, 0],[0, 0.01]])

A = np.array([[0, 1],[-1.6, -0.4]])
B = np.array([[0],[1]])

x0 = np.array([[10],[0]])

def sys(x, u):
    return np.dot(A, x) + np.dot(B, u)


#Solve Riccati
phi = linalg.solve_continuous_are(A, B, Q, R) #infinite, has to be finite!

#Compute response
# u(t) = -K(t)*x(t), K(t)=R_inv(t)*B.T(t)*Phi(t)
n = 200
T = 10
time_discr = np.linspace(0, T, n)
u = np.array([])
x = x0
#print("x= ", x)

for t in range(len(time_discr)-1):
    K = -1/R * np.dot(B.T, phi)
    ut = -np.dot(K, x[:,t])
    u = np.append(u, ut)


    x1n = T/n * x[1,t] 
    x2n = T/n * (-1.6*x[0,t] -0.4*x[1,t] + u[t])
    xn = np.array([[x1n],[x2n]])
    x = np.hstack((x, xn))


#Plots
fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.step(time_discr, x[0,:], label='x1')
ax1.step(time_discr, x[1,:], label='x2')
ax2.step(time_discr[0:-1], u, label='u')
ax1.legend()
ax2.legend()
ax2.set_xlabel('Time')
fig.suptitle('Riccati response')
plt.show()

#TODO: Plot difference x, u of responses riccati & TPVBP