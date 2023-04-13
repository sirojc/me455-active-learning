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

def F(t, c):
    return np.array([10*c[1]**2 + 3.2*c[1] - 2,
                    10*c[1]*c[2] - c[0] + 0.4*c[1] + 1.6*c[2],
                    10*c[2]**2 + 0.8*c[2] - 2*c[1] - 0.01])

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
t = tT+h
print("Computing Riccati...")
while t >= 0:
    k1 = F(t, c)
    k2 = F(t + h/2, c + h/2*k1)
    k3 = F(t + h/2, c + h/2*k2)
    k4 = F(t + h, c + h*k3)

    c = c + h/6 * (k1 + 2*k2 + 2*k3 + k4)
    i = int(t/tT * n)
    p1[i] = c[0]
    p2[i] = c[1]
    p3[i] = c[2]
    t = t + h

# u = -1/R*B.T*Phi(t)*x(t)
u = np.zeros(n)
x1 = np.zeros(n)
x1[0] = 10
x2 = np.zeros(n)
print("Computing states & input...")
for t in range(n-1):
    u[t] = -1/R *(p2[t]*x1[t] + p3[t]*x2[t])
    x1[t+1] = x1[t] + np.abs(h)*(x2[t])
    x2[t+1] = x2[t] + np.abs(h)*(-1.6*x1[t] -0.4*x2[t] + u[t])


#Plots
fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.step(time_discr, x1, label=r'$x_{1}$')
ax1.step(time_discr, x2, label=r'$x_{2}$')
ax2.step(time_discr, u, color='red', label=r'$u$')
ax1.legend()
ax2.legend()
ax2.set_xlabel('Time')
fig.suptitle('Response using Riccati with Runge-Kutta')

'''
fig3, (ax1, ax2, ax3) = plt.subplots(3, 1)
ax1.step(time_discr, p1)
ax2.step(time_discr, p2)
ax3.step(time_discr, p3)
ax2.set_xlabel('Time')
fig3.suptitle('Riccati Matrix')
'''

#Import results from Problem 2
res_P2 = np.genfromtxt("ME455_ActiveLearning/HW1/results_P2.csv", delimiter=',')
x1BP = res_P2[:,0]
x2BP = res_P2[:,1]
uBP = res_P2[:,2]

#Plot difference
fig2, (ax1, ax2) = plt.subplots(2, 1)
ax1.step(time_discr, x1BP - x1, label=r'$x_{1,BP} - x_{1,R}$')
ax1.step(time_discr, x2BP - x2, label=r'$x_{2,BP} - x_{2,R}$')
ax2.step(time_discr, uBP - u, color='red', label=r'$u_{BP} - u_{R}$')
ax1.legend()
ax2.legend()
ax1.set_ylabel(r'$\Delta~x(t)$')
ax2.set_ylabel(r'$\Delta~u(t)$')
ax2.set_xlabel('Time')
fig2.suptitle('Difference TPVBP vs. Riccati')
plt.savefig('ME455_ActiveLearning/HW1/Problem 3/HW1_3_diff.png')

plt.show()