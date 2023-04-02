import numpy as np
import matplotlib as plt
import scipy.integrate as intg

Q = np.array([[2, 0],[0, 0.01]])
R = 0.1
P = np.array([[1, 0],[0, 0.01]])

A = np.array([[0, 1],[-1.6, -0.4]])
B = np.array([[0],[1]])

x0 = np.array([10, 0])

def objective(x, u):
    integral = intg.quad(np.dot(np.dot(x.T, Q), x) + R*u**2, 0, 10)
    x10 = np.dot(np.dot(x[-1].T, P), x[-1]) # not index 9 probably!

    return 0.5*integral + 0.5*x10

def constraint(x, u):
    x.dt() = np.dot(B, x) + np.dot(C, u)
