import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg, integrate


Q = np.array([[2, 0],[0, 0.01]])
R = 0.1
P = np.array([[1, 0],[0, 0.01]])

A = np.array([[0, 1],[-1.6, -0.4]])
B = np.array([[0],[1]])

x0 = np.array([[10],[0]])



#Solve finite horizon Riccati:
# Phi' = Phi*B*R^-1*B.T*Phi - Phi*A - A.T*Phi - Q, Phi(T)=P

# Let Phi = a, Phi2 = b, Phi3 = c. One gets the following differential equations:
# a' = 10b^2 + 3.2c -2
# b' = 10bc -a + 0.4b + 1.6c
# c' = 10c^2 + 0.8c -2b -0.01

def F(t, c):
    return np.array([10*c[1]**2 + 3.2*c[2] - 2,
                    10*c[1]*c[2] - c[0] + 0.4*c[1] + 1.6*c[2],
                    10*c[2]**2 + 0.8*c[2] - 2*c[1] - 0.01])

t0 = 0
tT = 10
h = -0.01 # = 1000 steps; backwards, as condition for tT is given
n = tT*int(1/np.abs(h))
time_discr = np.linspace(t0, tT, n)

pT = np.array([1, 0, 0.01])

p1 = [pT[0]]
p2 = [pT[1]]
p3 = [pT[2]]

c = np.array([1, 0, 0.01])

# Implement Runge-Kutta
t = tT+h
print("Computing Riccati...")
while t > t0:
    c = c + h*F(t, c)

    p1 = np.insert(p1, 0, c[0])
    p2 = np.insert(p2, 0, c[1])
    p3 = np.insert(p3, 0, c[2])
    t = t + h

# u = -1/R*B.T*Phi(t)*x(t)
u = np.array([0])
x1 = np.array([10])
x2 = np.array([0])
print("Computing states & input...")
for t in range(n-1):
    u = np.append(u, -1/R *(p2[t]*x1[t] + p3[t]*x2[t]))
    x1 = np.append(x1, np.abs(h)*(x2[t]))
    x2 = np.append(x2, np.abs(h)*(-1.6*x1[t]-0.4*x2[t] + u[t]))


#Plots
fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.step(time_discr, x1, label='x1')
ax1.step(time_discr, x2, label='x2')
ax2.step(time_discr, u, label='u')
ax1.legend()
ax2.legend()
ax2.set_xlabel('Time')
fig.suptitle('Riccati response')


#Import results from Problem 2
'''
fig2, (ax1, ax2) = plt.subplots(2, 1)
ax1.step(time_discr, x1 - x1BP, label=r'$x1_{BP} - x1_{R}$')
ax1.step(time_discr, x2 - x2BP, label=r'$x2_{BP} - x2_{R}$')
ax2.step(time_discr, u - uBP, label=r'$u_{BP} - u_{R}$')
ax1.legend()
ax2.legend()
ax2.set_xlabel('Time')
fig.suptitle('Difference TPVBP vs. Riccati')
'''
plt.show()