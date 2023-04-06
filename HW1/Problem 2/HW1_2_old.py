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
ax1.step(time_discr, x1.value, label=r"$x1$")
ax1.step(time_discr, x2.value, label=r"$x2$")
ax2.step(time_discr, u.value, color='red', label=r"$u$")
ax1.set_ylabel(r"$x(t)$")
ax2.set_xlabel("Time")
ax2.set_ylabel(r"$u(t)$")
ax1.legend()
ax2.legend()
#plt.savefig('ME455_ActiveLearning/HW1/Problem 2/HW1_2_opt_time.png')

plt.show()



'''
#rewrite integral to sum!

def objective(x,t):
    u = 1.6(x + np.exp(-0.2*t)*(0.125+np.sin(0.8*t)-10*np.cos(0.8*t)))
    integral = quad(2*(x**2)+0.01*((x.dt())**2)+0.1*(u**2), 0, 10)
    add = x[-1]**2 + 0.01*((x[-1].dt())**2)
    return integral + add

con = ({'type': 'eq', 'fun': lambda x: x[0]==10})

res = minimize(objective, 10, constraints=con)
'''
