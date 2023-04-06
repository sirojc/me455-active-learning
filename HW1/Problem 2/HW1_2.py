import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import quad
from scipy.optimize import minimize
from gekko import GEKKO

#Objective & Dynamics
Q = np.array([[2, 0],[0, 0.01]])
R = 0.1
P = np.array([[1, 0],[0, 0.01]])

A = np.array([[0, 1],[-1.6, -0.4]])
B = np.array([[0],[1]])


#Time
T = 10
n = 100
dt = T / n
time_discr = np.linspace(0, T, n)

#y = [x1, x2, u]
y0 = np.array([10, 0, 0])
y = np.zeros((3, n))
initial_guess = np.zeros(3*n)
for e in range(n):
    initial_guess[e] = y0[0]
    initial_guess[n+e] = y0[1]
    initial_guess[2*n+e] = y0[2]

def objective(y):
    intg = 0
    for i in range(n):
        intg = intg + dt*(0.1*y[2*n+i]**2 + np.dot(np.dot([y[i], y[n+i]], Q), np.transpose([y[i], y[n+i]])))
    final = np.dot(np.dot([y[i], y[n+i]], P), np.transpose([y[i], y[n+i]]))
    return 0.5 * (intg + final)

tmp = []
tmp.append({'type': 'eq', 'fun': lambda y: y[0] - 10})
tmp.append({'type': 'eq', 'fun': lambda y: y[n] - 0})
for i in range(n-1):
        tmp.append({'type': 'eq', 'fun': lambda y: y[i+1] - y[i] - dt*(y[n+i])}) #y[i+1] == y[i] + dt*(y[n+i])
        tmp.append({'type': 'eq', 'fun': lambda y: y[n+i+1] - y[n+i] - dt*(-1.6*y[i] -0.4*y[n+i] + y[2*n + i])}) #y[n+i+1] == y[n+i] + dt*(-1.6*y[i] -0.4*y[n+i] + y[2*n + i])

cons = tuple(tmp)

print('Starting Optimization...')
res = minimize(objective, initial_guess, constraints=cons)
print(res)


'''
#Other try

x0 = np.array([10, 0])

m = GEKKO()

#Time
T = 10
n = 1000
time_discr = np.linspace(0, T, n)
m.time = time_discr

#Variables
x1 = m.Var(value=x0[0])
x2 = m.Var(value=x0[1])
u = m.Var()
t = m.Var(value=0, lb=0, ub=T)

#Constraints
m.Equation(x1.dt() == x2)
m.Equation(x2.dt() == -1.6*x1 -0.4*u) 
m.Equation(t.dt() == 1)

#Objective
#m.Obj(0.5*(m.integral(0.1*(u**2) + np.dot(np.dot(np.array([x1, x2]), np.array([[0, 1],[-1.6, -0.4]])), np.array([[x1],[x2]]))))) #Add final value!
m.Obj(0.5*(m.integral(0.1*(u**2) + 2*(x1**2) + 0.01*(x2**2))) + 0.5*(x1**2 + 0.01*(x2**2))) #Add final value!

#m.Obj(objective(x1, x2, u))

m.options.IMODE = 6
m.options.MAX_ITER=10000
print("Solving...")
m.solve(disp=True)

fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.step(time_discr, x1.value, label="x1")
ax1.step(time_discr, x2.value, label="x2")
ax2.step(time_discr, u.value, color='red', label="u")
ax1.set_ylabel("x(t)")
ax2.set_xlabel("Time")
ax2.set_ylabel("u(t)")
ax1.legend()
ax2.legend()
#plt.savefig('ME455_ActiveLearning/HW1/Problem 2/HW1_2_opt_time.png')

plt.show()
'''

