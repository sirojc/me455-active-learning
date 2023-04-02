import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

#plt.close('all')

x0 = [0, 0, np.pi/2-0.01] #x0[2]=np.pi/2 !!!
T = 2*np.pi
n = int(T/0.05)
time_discr = np.linspace(0, T, n)

'''
Goal Trajectory
'''
xg = []
for t in range(n):
    xg = np.append(xg, 2*time_discr[t] / np.pi)
yg = np.zeros(n)
thg = np.zeros(n)
thg.fill(90)

'''
Initial Trajectory
'''
def s_t(t):
    th = x0[2] - 0.5*t
    x = -2*np.cos(-0.5*t) +2
    y = -2*np.sin(-0.5*t)
    return x, y, th

s_init = {}
s_init[0] = []
s_init[1] = []
s_init[2] = []
for t in time_discr:
    x, y, th = s_t(t)
    s_init[0] = np.append(s_init[0], x)
    s_init[1] = np.append(s_init[1], y)
    s_init[2] = np.append(s_init[2], th)

#Plot
fig, (ax1, ax2) = plt.subplots(2, 1, height_ratios=[2, 1])
ax1.plot(s_init[0], s_init[1], label='Initial pos')
ax1.plot(xg, yg, color='red', label='Goal pos')
ax2.plot(time_discr, s_init[2]*180/np.pi, label="\u03B8 Initial")
ax2.plot(time_discr, thg, color='red', label="\u03B8 Goal")

ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_ylim([-2, 2])
ax2.set_xlabel("Time")
ax2.set_ylabel("Heading [°]")
ax1.legend()
ax2.legend()

fig.suptitle("Initial Trajectory")


'''
Optimized Trajectory
'''

from gekko import GEKKO
m = GEKKO()

#Time
m.time = np.linspace(0, T, n)

a = 1000
#Variables
x = m.Var(value=x0[0])
y = m.Var(value=x0[1])
th = m.Var(value=x0[2])
u1 = m.Var(value=0)#, lb=-a, ub=a)
u2 = m.Var(value=0)#, lb=-a, ub=a)
t = m.Var(value=0)#, lb=0, ub=T)

#Constraints
m.Equation(x.dt() == m.cos(th)*u1)
m.Equation(y.dt() == m.sin(th)*u1) 
m.Equation(th.dt() == u2)
m.Equation(t.dt() == 1)

#Objective
m.Obj(1*(x - 2/np.pi *t)**2 + 1*(y - 0)**2 + 1*(th - np.pi/2)**2)

m.options.IMODE = 6
m.options.MAX_ITER=10000
print("Solving...")
m.solve(disp=True)

costh = []
for t in range(n):
    costh = np.append(costh, np.cos(th.value[t]))
for t in range(n):
    th.value[t] = th.value[t] * 180/np.pi

#Plots
fig2, (ax1, ax2, ax3) = plt.subplots(3, 1)
ax1.step(time_discr, x.value, label="x")
ax2.step(time_discr, y.value, label="y")
ax1.plot(time_discr, xg, color='red', label="xg")
ax2.plot(time_discr, yg, color='red', label="yg")
ax3.step(time_discr, th.value, label="\u03B8")
ax3.plot(time_discr, thg, color='red', label="\u03B8g")
ax1.set_xlabel("Time")
ax1.set_ylabel("x")
ax2.set_xlabel("Time")
ax2.set_ylabel("y")
ax3.set_xlabel("Time")
ax3.set_ylabel("Heading [°]")
ax1.legend()
ax2.legend()
ax3.legend()
fig2.suptitle("Optimized Trajectory")

fig3, ax = plt.subplots()
ax.step(x.VALUE, y.VALUE, label='Optimized')
ax.plot(xg, yg, color='red', label='Goal')
ax.legend()
ax.set_ylim([-2, 2])
ax.set_xlabel("x")
ax.set_ylabel("y")
fig3.suptitle("Optimized Trajectory")

fig4, (ax1, ax2) = plt.subplots(2, 1)
ax1.step(time_discr, u1.value, label="u1")
ax2.step(time_discr, u2.value, color='orange', label="u2")
ax1.legend()
ax2.legend()
ax2.set_xlabel("Time")
ax1.set_ylabel("u1")
ax2.set_ylabel("u2")
fig4.suptitle("Optimized Controls")

plt.show()













'''
def objective():
    for i in range(len(time_discr)):
        tmp = (x[0][i] - xg[0][i])**2 (x[1][i] - xg[1][i])**2 (x[2][i] - xg[2][i])**2 # X has to be a 1-D array!!!
        sum = sum + tmp
        return sum
'''

'''
from gekko import GEKKO
m = GEKKO()
#Initialize variables
x = m.Array(m.Var,(3, 200))
for j in range(200):
        x[0,j].value = 0
        x[1,j].value = 0
        x[2,j].value = np.pi/2
u = m.Array(m.Var,(2,200))
#Equations
m.Equation(x[0,:].dt() == m.cos(x[2,:])*u[0,:]) #evtl. etwas wie [0,:]?
m.Equation(x[1,:].dt() == m.sin(x[2,:])*u[0,:]) 
m.Equation(x[2,:].dt() == u[1,:])
#m.Obj((x - 2/np.pi *t)**2 + (y - 0)**2 + (th - np.pi/2)**2)
m.Obj(sum([(x[0,j] - xg[0,j])**2 + (x[1,j] - 0)**2 + (x[2,j] - np.pi/2)**2 for j in range(len(xg))]))
m.options.IMODE = 5
print("Solving...")
m.solve(disp=True, GUI=True)
'''


'''
from gekko import GEKKO
m = GEKKO()
#Initialize variables
u1 = m.Var(0)
u2 = m.Var(0)
x = m.Var()
y = m.Var()
th = m.Var(value = np.pi/2)
t = m.Var(value = 0, lb = 0, ub = 2*np.pi)
#Equations
m.Equation(x.dt() == m.cos(th)*u1)
m.Equation(y.dt() == m.sin(th)*u1) 
m.Equation(th.dt() == u2)
m.Equation(t.dt() == 1)
m.Obj((x - 2/np.pi *t)**2 + (y - 0)**2 + (th - np.pi/2)**2)
m.options.IMODE = 5
print("Solving...")
m.solve(disp=True, GUI=True)
'''
'''
#x = [x, y, th, t, u1, u2]
def objective(x):
    return (x[0] - 2/np.pi *x[3])**2 + (x[1] - 0)**2 + (x[2] - np.pi/2)**2

x0 = np.array([0, 0, np.pi/2, 0, 0, 0])

cons = ({'type': 'eq', 'fun': lambda x: x[0].dt() == np.cos(x[2])*x[4]},
        {'type': 'eq', 'fun': lambda x: x[1].dt() == np.sin(x[2])*x[4]},
        {'type': 'eq', 'fun': lambda x: x[2].dt() == x[5]},
        {'type': 'eq', 'fun': lambda x: x[3].dt() == 1})

bds = ((None, None), (None, None), (None, None), (0, 2*np.pi), (None, None), (None, None),)

res = opt.minimize(objective, x0, method='SLSQP', bounds=bds, constraints=cons)
print(res.x)
'''
'''
x0 = np.array([0, 0, np.pi/2, 1, 1])
for t in time_discr:
    #x = [x, y, th, u1, u2]
    def objective(x):
        return (x[0] - 2/np.pi *t)**2 + (x[1] - 0)**2 + (x[2] - np.pi/2)**2
    
    cons = ({'type': 'eq', 'fun': lambda x: x[0] == (np.cos(x[2])*x[3])*T/n + x0[0]},
        {'type': 'eq', 'fun': lambda x: x[1] == (np.sin(x[2])*x[3])*T/n + x0[1]},
        {'type': 'eq', 'fun': lambda x: x[2] == x[4]*T/n + x0[2]})
    
    res = opt.minimize(objective, x0[-1], method='SLSQP', constraints=cons)
    x0 = [x0, res]
'''

'''
Latest try:
#cons = ({'type': 'eq', 'fun': lambda x: x[0] == (np.cos(x[2])*x[3])*T/n + x0[0]},
#        {'type': 'eq', 'fun': lambda x: x[1] == (np.sin(x[2])*x[3])*T/n + x0[1]},
#        {'type': 'eq', 'fun': lambda x: x[2] == x[4]*T/n + x0[2]})


res = opt.minimize(objective, x0)#, method='SLSQP', constraints=cons)
print(res)

'''