import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import quad
from scipy.optimize import minimize
from gekko import GEKKO

'''
Q = np.array([[2, 0],[0, 0.01]])
R = 0.1
P = np.array([[1, 0],[0, 0.01]])

A = np.array([[0, 1],[-1.6, -0.4]])
B = np.array([[0],[1]])

x0 = np.array([10, 0])
'''


'''
Other try
'''
x0 = np.array([10, 0])

m = GEKKO()

#Time
T = 10
n = 1000
h = T/n
time_discr = np.linspace(0, T, n)

#Variables
'''
x1 = [m.Var(10) for i in range(n)]
x2 = [m.Var(0) for i in range(n)]
u = [m.Var(0) for i in range(n)]
'''
x1 = m.Array(m.Var, n)
x2 = m.Array(m.Var, n)
u = m.Array(m.Var, n)

for i in range(n-1):
    x1[i].value = 10
    x2[i].value = 0
    u[i].value = 0


#Initial Conditions
m.Equation(x1[0]==10)
m.Equation(x2[0]==0) 

# Dynamic Constraints
for i in range(n-1):
    m.Equation((x1[i+1] - x1[i])/h == x2[i])
    m.Equation((x2[i+1] - x2[i])/h == -1.6*x1[i] -0.4*x2[i] + u[i])

#Objective
#m.Obj(0.5*(m.integral(0.1*(u**2) + np.dot(np.dot(np.array([x1, x2]), np.array([[0, 1],[-1.6, -0.4]])), np.array([[x1],[x2]]))))) #Add final value!

m.Obj(0.5*(x1[n-1]**2 + 0.01*x2[n-1]**2 + (m.sum([h*(0.1*u[i]**2 + 2*x1[i]**2 + 0.01*x2[i]**2) for i in range(n-1)]))))

m.options.IMODE = 3
m.options.MAX_ITER=10000
print("Solving...")
m.solve(disp=True)

x1f = []
x2f = []
uf = []
for i in range(n):
    x1f.append(x1[i].value)
    x2f.append(x2[i].value)
    uf.append(u[i].value)


fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.step(time_discr, x1f, label=r"$x1$")
ax1.step(time_discr, x2f, label=r"$x2$")
ax2.step(time_discr, uf, color='red', label=r"$u$")
ax1.set_ylabel(r"$x(t)$")
ax2.set_xlabel("Time")
ax2.set_ylabel(r"$u(t)$")
ax1.legend()
ax2.legend()
plt.savefig('ME455_ActiveLearning/HW1/Problem 2/HW1_2_opt_time.png')

plt.show()
