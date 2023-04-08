import numpy as np
import matplotlib.pyplot as plt

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

t0 = 0
tT = 10
h = -0.01 # = 1000 steps; backwards, as condition for tT is given
n = tT*int(1/np.abs(h))
time_discr = np.linspace(t0, tT, n)

pT = np.array([1, 0, 0.01])

p1 = np.zeros(n)
p2 = np.zeros(n)
p3 = np.zeros(n)
p1[n-1] = pT[0]
p2[n-1] = pT[1]
p3[n-1] = pT[2]

c = np.array([1, 0, 0.01])

# Implement Runge-Kutta
i = n-2
print("Computing Riccati.")
while i >= 0:
    p1[i] = p1[i+1] + h*(10*p2[i+1]**2 + 3.2*p3[i+1] - 2)
    p2[i] = p2[i+1] + h*(10*p2[i+1]*p3[i+1] - p1[0] + 0.4*p2[i+1] + 1.6*p3[i+1])
    p3[i] = p3[i+1] + h*(10*p3[i+1]**2 + 0.8*p3[i+1] - 2*p2[i+1] - 0.01)
    i = i -1

# u = -1/R*B.T*Phi(t)*x(t)
u = np.zeros(n)
x1 = np.zeros(n)
x1[0] = 10
x2 = np.zeros(n)
print("Computing states & input.")
for i in range(n-1):
    u[i] = -1/R *(p2[i]*x1[i] + p3[i]*x2[i])
    x1[i+1] = x1[i] + np.abs(h)*(x2[i])
    x2[i+1] = x2[i] + np.abs(h)*(-1.6*x1[i] -0.4*x2[i] + u[i])


#Plots
fig2, (ax1, ax2, ax3) = plt.subplots(3, 1)
ax1.step(time_discr, p1)
ax2.step(time_discr, p2)
ax3.step(time_discr, p3)
ax2.set_xlabel('Time')
fig2.suptitle('Riccati')

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