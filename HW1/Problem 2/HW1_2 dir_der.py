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
a = 1
b = 1
c = 0

for i in range(10):
    dir[i] = np.zeros((3, n))
    val[0].append(round(a, 3))
    val[1].append(round(b, 3))
    val[2].append(round(c, 3))
    for t in range(n-1):
        dir[i][2,t] = a * np.sin(2*np.pi/n * t * b) + c
        dir[i][0,t+1] = dir[i][0,t] + h*dir[i][1,t]
        dir[i][1,t+1] = dir[i][1,t] + h*(-1.6*dir[i][0,t] -0.4*dir[i][1,t] + dir[i][2,t]) 
        abs = np.sqrt(dir[i][0,t]**2 + dir[i][1,t]**2 + dir[i][2,t]**2)
    a *= 1.3
    b *= -1.1
    c += (0.2+0.5*c)

dir_der = np.zeros(10)

# Calculate directional derivative
for z in range(len(dir)):
    intg = 0
    for t in range(n-1):
        intg += h*(2*x1sol[t]*dir[z][0,t] + 0.01*x2sol[t]*dir[z][1,t] + 0.1*usol[t]*dir[z][2,t])
    fV = x1sol[-1]*dir[z][0,-1] + 0.01*x2sol[-1]*dir[z][1,-1]
    dir_der[z] = round(intg + fV,9)

#Generate a table figure showing the index and dir_der in a two column table
fig, ax = plt.subplots(figsize=(4, 3))
ax.axis('tight')
ax.axis('off')
the_table = ax.table(cellText=[['Idx','a','b','c', '$DJ(x_{sol}(t), u_{sol}(t))*$\u03B6$(t)$'], ['1',val[0][0], val[1][0], val[2][0], dir_der[0]],
                               ['2',val[0][1], val[1][1], val[2][1], dir_der[1]],
                               ['3',val[0][2], val[1][2], val[2][2], dir_der[2]],
                               ['4',val[0][3], val[1][3], val[2][3], dir_der[3]],
                               ['5',val[0][4], val[1][4], val[2][4], dir_der[4]],
                               ['6',val[0][5], val[1][5], val[2][5], dir_der[5]],
                               ['7',val[0][6], val[1][6], val[2][6], dir_der[6]],
                               ['8',val[0][7], val[1][7], val[2][7], dir_der[7]],
                               ['9',val[0][8], val[1][8], val[2][8], dir_der[8]],
                               ['10',val[0][9], val[1][9], val[2][9], dir_der[9]]], loc='center')
ax.set_title(r"$\frac{d}{dt}z(t) = Az + Bv, v(t) = a*sin(\frac{2pi*b}{n}t) + c$")
the_table.auto_set_column_width(col=0)
the_table.auto_set_column_width(col=1)
the_table.auto_set_column_width(col=2)
the_table.auto_set_column_width(col=3)
the_table.auto_set_column_width(col=4)
plt.savefig('ME455_ActiveLearning/HW1/Problem 2/HW1_2_table.png')
plt.show()

