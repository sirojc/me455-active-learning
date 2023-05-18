import numpy as np
import matplotlib.pyplot as plt

# Plots LaTeX-Style
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'

def Phi(x):
    return np.linalg.det(2*np.pi*Sig)**(-1/2) * np.exp(-1/2 * np.dot(np.dot((x-mu).T, np.linalg.inv(Sig)), (x-mu)))


def main():
    b = 0
    T = 100

    A = np.array([[0, 1],[-1, -b]])
    x0 = np.array([0, 1])

    Sig = np.diag([2, 2])
    mu = 0

if __name__ == "__main__":
    main()