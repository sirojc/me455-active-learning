import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

#plt.close('all')

x0 = [0, 0, np.pi/2-0.01] #x0[2]=np.pi/2 !!!
T = 2*np.pi
n = 1000 # int(T/0.05)
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
fig, (ax1, ax2) = plt.subplots(2, 1, height_ratios=[2, 1], figsize=(6, 6))
ax1.plot(s_init[0], s_init[1], label='Initial pos')
ax1.plot(xg, yg, color='red', label='Goal pos')
ax2.plot(time_discr, s_init[2]*180/np.pi, label="\u03B8 Initial")
ax2.plot(time_discr, thg, color='red', label="\u03B8 $_{d}$")

ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_ylim([-2, 2.1])
ax2.set_xlabel("Time")
ax2.set_ylabel("\u03B8(t) [°]")
ax1.legend()
ax2.legend()

fig.suptitle("Initial Trajectory")
plt.savefig('ME455_ActiveLearning/HW1/Problem 1/HW1_1_init_traj.png')


'''
Optimized Trajectory
'''

from gekko import GEKKO
m = GEKKO()

#Time
m.time = np.linspace(0, T, n)

a = 5
#Variables
x = m.Var(value=x0[0])
y = m.Var(value=x0[1])
th = m.Var(value=x0[2])
u1 = m.Var(value=0, lb=-a, ub=a)
u2 = m.Var(value=0, lb=-a, ub=a)
t = m.Var(value=0, lb=0, ub=T)

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
ax1.plot(time_discr, xg, color='red', label=r"$x_{d}$")
ax2.plot(time_discr, yg, color='red', label=r"$y_{d}$")
ax3.step(time_discr, th.value, label="\u03B8")
ax3.plot(time_discr, thg, color='red', label="\u03B8$_{d}$")
ax1.set_xlabel("Time")
ax1.set_ylabel("x(t)")
ax2.set_xlabel("Time")
ax2.set_ylabel("y(t)")
ax3.set_xlabel("Time")
ax3.set_ylabel("\u03B8(t) [°]")
ax1.legend()
ax2.legend()
ax3.legend()
fig2.suptitle("Optimized Trajectory")
plt.savefig('ME455_ActiveLearning/HW1/Problem 1/HW1_1_opt_time.png')

fig3, ax = plt.subplots()
ax.step(x.VALUE, y.VALUE, label='Optimized')
ax.plot(xg, yg, color='red', label='Goal')
ax.legend()
ax.set_ylim([-2, 2])
ax.set_xlabel("x")
ax.set_ylabel("y")
fig3.suptitle("Optimized Trajectory")
plt.savefig('ME455_ActiveLearning/HW1/Problem 1/HW1_1_opt_xy.png')

fig4, (ax1, ax2) = plt.subplots(2, 1)
ax1.step(time_discr, u1.value, label=r"u$_{1}$")
ax2.step(time_discr, u2.value, color='orange', label=r"$u_{2}$")
ax1.legend(loc='upper right')
ax2.legend(loc='upper right')
ax2.set_xlabel("Time")
ax1.set_ylabel(r"$u_{1}(t)$")
ax2.set_ylabel(r"$u_{2}(t)$")
fig4.suptitle("Optimized Controls")
plt.savefig('ME455_ActiveLearning/HW1/Problem 1/HW1_1_opt_u.png')

plt.show()