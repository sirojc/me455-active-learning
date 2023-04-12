import numpy as np
from matplotlib import pyplot as plt

res_P2 = np.genfromtxt("ME455_ActiveLearning/HW1/results_P2.csv", delimiter=',')
x1sol = res_P2[:,0]
x2sol = res_P2[:,1]
usol = res_P2[:,2]

T = 10
n = 1000
h = T/n

# Initialize directions at every timesteps
dir = {}
a = 1
b = 1
c = 1
d = 1
for i in range(10):
    dir[i] = np.zeros((3, n))
    for t in range(n-1):
        dir[i][0,t] = np.sin(2*np.pi/n * t * a)
        dir[i][1,t] = np.cos(2*np.pi/n * t * b)
        dir[i][2,t] = d * np.sin(2*np.pi/n * t * c)
        abs = np.sqrt(dir[i][0,t]**2 + dir[i][1,t]**2 + dir[i][2,t]**2)
        dir[i][:,t] /= abs # nowhere stated that it has to be normalized
    a *= 1.2
    b *= 1.3
    c *= 1.6
    d *= -0.8

dir_der = np.zeros(10)

# Calculate directional derivative
for z in range(len(dir)):
    intg = 0
    for i in range(n-1):
        intg += h*(2*x1sol[i]*dir[z][0,i] + 0.01*x2sol[i]*dir[z][1,i] + 0.1*usol[i]*dir[z][2,i])
    fV = x1sol[-1]*dir[z][0,-1] + 0.01*x2sol[-1]*dir[z][1,-1]
    dir_der[z] = intg + fV

print(dir_der)