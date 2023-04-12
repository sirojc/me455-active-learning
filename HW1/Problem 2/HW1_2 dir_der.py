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
val = {}
val[0] = []
val[1] = []
val[2] = []
val[3] = []
a = 1
b = 1
c = 1
d = 1

for i in range(10):
    dir[i] = np.zeros((3, n))
    val[0].append(round(a, 3))
    val[1].append(round(b, 3))
    val[2].append(round(c, 3))
    val[3].append(round(d, 3))
    for t in range(n-1):
        dir[i][0,t] = np.sin(2*np.pi/n * t * a) # Show that vector is normalized above table
        dir[i][1,t] = np.cos(2*np.pi/n * t * b)
        dir[i][2,t] = d * np.sin(2*np.pi/n * t * c)
        abs = np.sqrt(dir[i][0,t]**2 + dir[i][1,t]**2 + dir[i][2,t]**2)
        dir[i][:,t] /= abs # nowhere stated that it has to be normalized, Wikipedia not sure
    a *= 1.2
    b *= 1.3
    c *= 1.6
    d *= -0.8

dir_der = np.zeros(10)

# Calculate directional derivative
for z in range(len(dir)):
    intg = 0
    for t in range(n-1):
        intg += h*(2*x1sol[t]*dir[z][0,t] + 0.01*x2sol[t]*dir[z][1,t] + 0.1*usol[t]*dir[z][2,t])
    fV = x1sol[-1]*dir[z][0,-1] + 0.01*x2sol[-1]*dir[z][1,-1]
    dir_der[z] = intg + fV

print(dir_der)

#Generate a table figure showing the index and dir_der in a two column table
fig, ax = plt.subplots()
ax.axis('tight')
ax.axis('off')
the_table = ax.table(cellText=[['a','b','c','d', 'Directional Derivative'], [val[0][0], val[1][0], val[2][0], val[3][0], dir_der[0]],
                               [val[0][1], val[1][1], val[2][1], val[3][1], dir_der[1]],
                               [val[0][2], val[1][2], val[2][2], val[3][2], dir_der[2]],
                               [val[0][3], val[1][3], val[2][3], val[3][3], dir_der[3]],
                               [val[0][4], val[1][4], val[2][4], val[3][4], dir_der[4]],
                               [val[0][5], val[1][5], val[2][5], val[3][5], dir_der[5]],
                               [val[0][6], val[1][6], val[2][6], val[3][6], dir_der[6]],
                               [val[0][7], val[1][7], val[2][7], val[3][7], dir_der[7]],
                               [val[0][8], val[1][8], val[2][8], val[3][8], dir_der[8]],
                               [val[0][9], val[1][9], val[2][9], val[3][9], dir_der[9]]], loc='center')
the_table.auto_set_column_width(col=0)
the_table.auto_set_column_width(col=1)
the_table.auto_set_column_width(col=2)
the_table.auto_set_column_width(col=3)
plt.show()
