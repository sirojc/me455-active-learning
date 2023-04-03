import numpy as np
import matplotlib as plt
from scipy.integrate import quad
from scipy.optimize import minimize

'''
Q = np.array([[2, 0],[0, 0.01]])
R = 0.1
P = np.array([[1, 0],[0, 0.01]])

A = np.array([[0, 1],[-1.6, -0.4]])
B = np.array([[0],[1]])

x0 = np.array([10, 0])
'''

#rewrite integral to sum!

def objective(x,t):
    u = 1.6(x + np.exp(-0.2*t)*(0.125+np.sin(0.8*t)-10*np.cos(0.8*t)))
    integral = quad(2*(x**2)+0.01*((x.dt())**2)+0.1*(u**2), 0, 10)
    add = x[-1]**2 + 0.01*((x[-1].dt())**2)
    return integral + add

con = ({'type': 'eq', 'fun': lambda x: x[0]==10})

res = minimize(objective, 10, constraints=con)