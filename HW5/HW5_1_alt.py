import numpy as np
import matplotlib.pyplot as plt

import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

from dataclasses import dataclass
from scipy.integrate import dblquad as integrate
from cachetools import cached

def printProgressBar(iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

@dataclass(frozen=True)
class Params:
    """Essential parameters for given problem"""
    T: float = 10
    dt: float = 0.1

    x_0: float = 0
    y_0: float = 1
    u_0 = np.array([0.2, -0.2])

    q: float = 10.
    Q = np.diag([1.,1.]) # np.diag([0.1, 0.1])
    R = np.diag([0.01,0.01]) # np.diag([0.1, 0.1])
    M = np.diag([0,0]) # no terminal cost

    alpha: float = 0.1
    beta: float = 0.5
    eps: float = 1E-3
    max_iterations: int = 200

    K: int = 10
    lb, ub = (-3.,3.)
    mu = np.array([0.0,0.0], dtype=float)
    Sigma = np.diag([2.,2.])


class State:
    def __init__(self, xytheta, t: float = 0):
        if isinstance(xytheta, np.ndarray):
            assert xytheta.size == 2 , f"State has wrong size, expected 2, got: {xytheta.size}"
            xytheta = xytheta.reshape(2)
            self.x = xytheta[0]
            self.y = xytheta[1]
        elif isinstance(xytheta, list) or isinstance(xytheta, tuple):
            self.x = xytheta[0]
            self.y = xytheta[1]
        else:
            raise ValueError(f"unsupported state format: {type(xytheta)}")
        self.t = t

    def __call__(self):
        return np.array([self.x, self.y])

    def dynamics(self, U: np.ndarray) -> np.ndarray:
        """Dynamics of a simple two-wheeler"""
        assert U.size == 2 , f"Input U has wrong size, expected 2, got: {U.size}"
        xdot: float = U[0]
        ydot: float = U[1]
        return np.array([xdot, ydot])
    
    def next(self, U: np.ndarray, dt: float = Params.dt):
        """Computes next state from current state with given input U using Euler integration"""
        assert U.size == 2 , f"Input U has wrong size, expected 2, got: {U.size}"
        next_state: np.ndarray = self() + dt * self.dynamics(U)
        return State(next_state, self.t + dt)
    

class StatePertubation(State):
    def __init__(self, xytheta, t: float = 0):
        super().__init__(xytheta, t)
    
    def dynamics(self, v: np.ndarray) -> np.ndarray:
        """Linearized pertubation dynamics in D_1f/D1_f resp. D_2f/D2_f"""
        A = np.array([0, 0,\
                      0, 0,]).reshape(2,2)    # D_1f
        B = np.array([1, 0, 0, 1]).reshape(2,2) # D_2f
        x_dot = np.dot(A, self()) + np.dot(B, v)
        return x_dot
    
    def next(self, v: np.ndarray, dt: float = Params.dt):
        """Computes next state from current state with given input U using Euler integration"""
        assert v.size == 2 , f"Input U has wrong size, expected 2, got: {v.size}"
        next_state: np.ndarray = self() + dt * self.dynamics(v)
        return StatePertubation(next_state, self.t + dt)


class ErgodicControl:
    def __init__(self, T: float = Params.T, dt: float = Params.dt, 
                 K: int = Params.K, lb: float = Params.lb, ub: float = Params.ub) -> None:
        self.T = T
        self.dt = dt
        self.N = int(np.ceil(T/dt))
        self.K = [(i, j) for i in range(K+1) for j in range(K+1)]
        self.hk = [1.0 for i in range(K+1) for j in range(K+1)]

        # gaussian distribution init
        self.mu = Params.mu
        self.Sigma = Params.Sigma
        self.Sigma_inv = np.linalg.inv(self.Sigma)
        self.normalizer = ( np.linalg.det( 2*np.pi*self.Sigma ) )**-.5

        self.lb = lb
        self.ub = ub

        self.K, self.Phi_K = self.get_spatial_distro_coeffs()
        _, self.Lambda_K = self.get_lambda_cofficients()
        self.hk = self.normalize_basis_function()
        return
    
    def normal_dist(self, x: np.ndarray = None):
        """Returns density of Gaussian Distribution of a given point x in R^2"""
        assert x.shape == self.mu.shape
        prob_density: float = self.normalizer *\
            np.exp( -0.5 * np.transpose(x - self.mu)@self.Sigma_inv@(x - self.mu) )
        return prob_density
        
    def normalize_basis_function(self):
        # return self.hk
        lb = self.lb
        ub = self.ub
        K = self.K
        hk = self.hk
        for idx, k in enumerate(K):
            integrand = lambda x1, x2: (np.cos(k[0]*np.pi/(ub-lb) * (x1 - ub)) * np.cos(k[1]*np.pi/(ub-lb) * (x2 - ub)))**2
            hk[idx] = np.sqrt(integrate(integrand, lb, ub, lb, ub)[0])
        return hk
    
    def get_basis_function(self, x: np.ndarray, k: tuple[int,int]):
        # F_k = self.normalize_basis_function(k)
        F_k = self.hk[k[0] * Params.K + k[1]]
        for dim in range(x.shape[-1]):
            F_k *= np.cos( k[dim] * (x[dim]-self.ub) * np.pi / (self.lb - self.ub) )
        return F_k
    
    def get_basis_deriv(self, x: np.ndarray, k: tuple[int,int]):
        # dF_k = np.array([self.normalize_basis_function(k), self.normalize_basis_function(k)]) / Params.T
        idx = k[0] * Params.K + k[1]
        dF_k = np.array([self.hk[idx], self.hk[idx]]) / Params.T

        dF_k[0] *= - np.sin( k[0] * (x[0]-self.ub) * np.pi / (self.lb - self.ub) ) 
        dF_k[0] *= k[0] * np.pi / (self.lb - self.ub)
        dF_k[0] *= np.cos( k[1] * (x[1]-self.ub) * np.pi / (self.lb - self.ub) )

        dF_k[1] *= - np.sin( k[1] * (x[1]-self.ub) * np.pi / (self.lb - self.ub) ) 
        dF_k[1] *= k[1] * np.pi / (self.lb - self.ub)
        dF_k[1] *= np.cos( k[0] * (x[0]-self.ub) * np.pi / (self.lb - self.ub) )

        return dF_k
        
    def get_fourier_coeffs(self, state_trajectory: np.ndarray[State]):
        c_k = 1/self.T
        K = self.K
        coefficients = [None] * len(K)
        for idx,k in enumerate(K):
            integrator = [c_k*self.get_basis_function(_x(), k) for _x in state_trajectory]
            coefficients[idx] = np.trapz(integrator, dx=self.dt)
        return K, coefficients
    
    def get_fourier_diff(self, state_trajectory, state_pertubation):
        c_k = 1.0
        K = self.K
        coefficients = [None] * len(K)
        for idx,k in enumerate(K):
            integrator = [c_k*self.get_basis_deriv(_x(), k)@_z() for _x, _z in zip(state_trajectory, state_pertubation)]
            coefficients[idx] = np.trapz(integrator, dx=self.dt)
        return K, coefficients
        
    @cached(cache ={})
    def get_spatial_distro_coeffs(self):
        K = self.K
        l = len(K)
        coefficients = [None] * l
        for idx,k in enumerate(K):
            coefficients[idx], _ = integrate(lambda x2,x1:
                self.normal_dist(np.array([x1,x2])) * self.get_basis_function(np.array([x1,x2]),k),
                self.lb, self.ub, self.lb, self.ub)
            printProgressBar(idx + 1, l, prefix = '    Progress:', suffix = 'Complete', length = 50)

        return K, coefficients
    
    def get_lambda_cofficients(self):
        K = self.K
        l = len(K)
        coefficients = [None] * l
        s = (2+1)/2
        for idx,k in enumerate(K):
            k_squared = k[0]**2 + k[-1]**2
            coefficients[idx] = (1 + k_squared)**(-s)
            
        return K, coefficients
    

def plot_optimized(state_trajectory: np.ndarray[State], input_trajectory: np.ndarray, initial_trajectory, U_0) -> None:
    time = [state.t for state in state_trajectory] # np.arange(0,state_trajectory.size)*Params.dt
    x = [state.x for state in state_trajectory]
    y = [state.y for state in state_trajectory]

    x_init = [state.x for state in initial_trajectory]
    y_init = [state.y for state in initial_trajectory]
    
    fig2, ax = plt.subplots()
    ax.plot(x_init, y_init, color='darkorange', label=r'$x_{0}(t)$')
    ax.plot(x, y, color='slateblue', label=r'$x_{erg}(t)$')
    ax.legend(loc='upper right')
    ax.set_xlabel(r'$x_{1}$')
    ax.set_ylabel(r'$x_{2}$')
    ax.set_aspect('equal')
    ax.set_title(r'Maximally ergodic trajectory')
    fig2.tight_layout()
    plt.savefig('./ME455_ActiveLearning/HW5/HW5_1_ergodic_traj.png', dpi=300)

    plt.show()
    return


def J(state_trajectory: np.ndarray[State], input_trajectory: np.ndarray, ergodic_metric: ErgodicControl) -> float:
    """Returns the cost of a given state-input trajectory"""
    assert input_trajectory.shape == (len(state_trajectory),2) ,  f"State/input has wrong size"

    dt = Params.dt
    q, Q, R, M = Params.q, Params.Q, Params.R, Params.M
    cost : float = 0

    _, C_K = ergodic_metric.get_fourier_coeffs(state_trajectory)
    ergodicity = sum([ l*(c-p)**2 for l,c,p in zip(ergodic_metric.Lambda_K, C_K, ergodic_metric.Phi_K) ])
    print("    Ergodic Metric is = ",ergodicity)
    cost += q * ergodicity

    for ii in range(len(state_trajectory)-1):
        u_curr = input_trajectory[ii]
        cost += dt * np.dot(u_curr,np.dot(R, u_curr))

    return cost


def Directional_J(state_trajectory: np.ndarray[State], input_trajectory: np.ndarray,\
                  state_pertubation: np.ndarray[StatePertubation], input_pertubation: np.ndarray, ergodic_metric: ErgodicControl) -> float:
    """Returns the directional derivative of the cost given an state-input trajectory and a state/input pertubation"""
    assert input_trajectory.shape == (len(state_trajectory),2) ,  f"State/input has wrong size"
    assert input_trajectory.shape == input_pertubation.shape,  f"Input/pertubation has wrong size"
    assert len(state_trajectory) == len(state_pertubation) ,  f"State/pertubation has wrong size"

    dt = Params.dt
    q, Q, R, M = Params.q, Params.Q, Params.R, Params.M
    cost : float = 0

    for ii in range(len(state_trajectory)-1):
        x_curr = state_trajectory[ii]
        u_curr = input_trajectory[ii]  

        d1l_curr = D1_l(x_curr, ergodic_metric, state_trajectory).reshape(1,2)
        z_curr = state_pertubation[ii]().reshape(2,1)
        v_curr = input_pertubation[ii]
        cost += (d1l_curr @ z_curr)[0,0]
        cost += dt * np.dot(v_curr,np.dot(R, u_curr))
        
    return cost # TODO


def D1_l(x_curr: State, ergodic_metric: ErgodicControl, state_trajectory: np.ndarray[State]) -> np.ndarray:
    q = Params.q
    x_curr = x_curr()

    _, C_K = ergodic_metric.get_fourier_coeffs(state_trajectory)

    K = ergodic_metric.K
    F_k = np.empty((len(K),2,))

    x = x_curr
    for idx,k in enumerate(K):
        F_k[idx] = ergodic_metric.get_basis_deriv(x, k)

    ergodicity = q * 2 * sum([ l*(c-p)*f for l,c,p,f in zip(ergodic_metric.Lambda_K, C_K, ergodic_metric.Phi_K, F_k) ])
    
    return ergodicity.reshape(2,1)


def D2_l(u_curr: np.ndarray) -> np.ndarray:
    R = Params.R
    return np.dot(R, u_curr).reshape(2,1) # DONE


def D1_f(x_curr: State, u_curr: np.ndarray) -> np.ndarray:
    matrix = np.array([0, 0,\
                      0, 0]).reshape(2,2)    # D_1f
    return matrix


def D2_f(x_curr: State) -> np.ndarray:
    matrix = np.array([1, 0, 0, 1]).reshape(2,2) # D_2f
    return matrix


def descent_direction(state_trajectory: np.ndarray[State], input_trajectory: np.ndarray, ergodic_metric: ErgodicControl) -> tuple:
    dt = Params.dt
    Q = Params.Q
    R = Params.R
    M = Params.M
    R_inv = np.linalg.inv(R)
    N = len(state_trajectory)
    P = np.zeros((N,2,2))
    r = np.zeros((N,2,1))
    A = np.zeros((N,2,2))
    B = np.zeros((N,2,2))

    P[-1] = M

    # load A,B
    for ii in range(N):
        A[ii] = D1_f(state_trajectory[ii],input_trajectory[ii])
        B[ii] = D2_f(state_trajectory[ii])
    
    # iterate through P
    print("    Calculating Descent Direction")
    for ii in range(N-1,0,-1):
        x_curr = state_trajectory[ii]
        u_curr = input_trajectory[ii]
        minus_Pdot = P[ii].dot(A[ii]) + np.transpose(A[ii]).dot(P[ii]) - \
                     P[ii].dot(B[ii]).dot(R_inv).dot(np.transpose(B[ii]).dot(P[ii])) + Q
        minus_rdot = np.transpose( A[ii] - B[ii].dot(R_inv).dot(np.transpose(B[ii]).dot(P[ii])) ).dot(r[ii]) +\
                     D1_l(x_curr, ergodic_metric, state_trajectory) - P[ii].dot(B[ii]).dot(R_inv.dot(D2_l(u_curr)))
        P[ii-1] = P[ii] + dt * minus_Pdot
        r[ii-1] = r[ii] + dt * minus_rdot
        printProgressBar(N - ii + 1, N, prefix = '    Progress:', suffix = 'Complete', length = 50)

    z_0 = np.array([0,0]) # -np.linalg.inv(P[0]).dot(r[0])
    state_pertubation = np.zeros_like(state_trajectory, dtype=StatePertubation)
    input_pertubation = np.zeros_like(input_trajectory)
    state_pertubation[0] = StatePertubation(z_0, t=0)

    for jj in range(N-1):
        input_pertubation[jj] -= (R_inv.dot(np.transpose(B[jj])).dot(P[jj]).dot(state_pertubation[jj]())).reshape(2)
        input_pertubation[jj] -= (R_inv.dot(np.transpose(B[jj])).dot(r[jj]) + R_inv.dot(D2_l(input_trajectory[jj]))).reshape(2)
        # z_next = dt * ((A[jj].dot(state_pertubation[jj]())).reshape(3) + B[jj].dot(input_pertubation[jj]))
        z_next = state_pertubation[jj]() + dt * ((A[jj].dot(state_pertubation[jj]())).reshape(2) + B[jj].dot(input_pertubation[jj]))
        # state_pertubation[jj+1] = state_pertubation[jj].next(input_pertubation[jj])
        state_pertubation[jj+1] = StatePertubation(z_next, (jj+1)*dt)
    
    input_pertubation[-1] -= (R_inv.dot(np.transpose(B[-1])).dot(P[-1]).dot(state_pertubation[-1]())).reshape(2)
    input_pertubation[-1] -= (R_inv.dot(np.transpose(B[-1])).dot(r[-1]) + R_inv.dot(D2_l(input_trajectory[-1]))).reshape(2)

    return (state_pertubation, input_pertubation)


def main() -> int:
    T: float = Params.T
    dt: float = Params.dt
    N = int(np.ceil(T/dt))
    alpha = Params.alpha
    beta = Params.beta
    eps = Params.eps
    max_iterations = Params.max_iterations

    # Initial trajectory
    # TODO Make some fancy Trajectory class

    X_0 = State((Params.x_0, Params.y_0), t=0)
    U_0 = np.kron( np.ones((N,1)), Params.u_0 )
    initial_trajectory: np.ndarray[State] = np.empty(N, dtype=State)
    initial_trajectory[0] = X_0
    for ii in range(N-1):
        U_0[ii] = [-0.5*np.cos(np.pi/50 *ii), -0.62*np.sin(np.pi/50 *ii)]
        initial_trajectory[ii+1] = initial_trajectory[ii].next(U_0[ii], dt)

    ergodic_metric = ErgodicControl()
    print("Calculating Spatial Distro Coefficients")

    initial_cost = J(initial_trajectory, U_0, ergodic_metric)
    initial_deriv = Directional_J(initial_trajectory, U_0, initial_trajectory, U_0, ergodic_metric)
    cost = []
    dcost = []

    current_state_trajectory = initial_trajectory
    current_input_trajectory = U_0

    z_0 = np.array([0,0])
    current_state_pertubation = np.zeros_like(current_state_trajectory, dtype=StatePertubation)
    current_input_pertubation = np.zeros_like(current_input_trajectory)
    current_state_pertubation[:] = StatePertubation(z_0, t=0)
    counter: int = 0
    while True:
        current_cost= J(current_state_trajectory, current_input_trajectory, ergodic_metric)
        current_dcost = Directional_J(current_state_trajectory, current_input_trajectory, current_state_pertubation, current_input_pertubation, ergodic_metric)
        cost.append(current_cost)
        dcost.append(current_dcost)
        print("    J = ", current_cost)
        print("    dJ = ",current_dcost)
        if counter > 0 and abs(Directional_J(current_state_trajectory, current_input_trajectory, \
                                         current_state_pertubation, current_input_pertubation, ergodic_metric)) <= eps:
            break
        if counter > max_iterations:
            break # no solution found
        
        current_state_pertubation, current_input_pertubation = descent_direction(current_state_trajectory, current_input_trajectory, ergodic_metric)

        n: int = 0
        gamma: float = 0.1
        while True:
            new_input_trajectory = current_input_trajectory + gamma * current_input_pertubation
            new_state_trajectory: np.ndarray[State] = np.empty(N, dtype=State)
            new_state_trajectory[0] = X_0
            for ii in range(N-1):
                new_state_trajectory[ii+1] = new_state_trajectory[ii].next(new_input_trajectory[ii], dt)

            if n > 0 and J(new_state_trajectory, new_input_trajectory, ergodic_metric) <\
                         J(current_state_trajectory, current_input_trajectory, ergodic_metric) +\
                          alpha * gamma * Directional_J(current_state_trajectory, current_input_trajectory,\
                                                      current_state_pertubation, current_input_pertubation, ergodic_metric):
                break
            if True or n > max_iterations: # TODO change
                gamma = 0.1
                break
                # return -1 # failed to converge
            
            
            # new_state_trajectory = X_0 + # HUH?
            n += 1
            gamma = beta**n
            print("    Current n:     ", n)
            print("    Current gamma: ", gamma)
            # end Armijo search

        current_input_trajectory = new_input_trajectory
        current_state_trajectory = new_state_trajectory

        counter += 1
        print("Current counter: ", counter)
        # end main while loop
    
    fig, axs = plt.subplots(2)
    axs[0].step(np.arange(len(cost)), cost, 'k', label=r'J$(\xi_i)$', color='darkorange')
    axs[0].set_xlabel(r'$i$')
    axs[0].set_title(r'Cost $J(\xi_i)$ vs. Iteration $i$')
    fig.tight_layout()
    
    axs[1].step(np.arange(len(dcost)-1), [abs(dcost_) for dcost_ in dcost[1:]], color='darkorange')
    axs[1].set_yscale('log')
    axs[1].set_xlabel(r'$i$')
    axs[1].set_title(r'$DJ(\xi_i)\cdot\zeta_{i}$ vs. Iteration $i$')
    plt.savefig('./ME455_ActiveLearning/HW5/HW5_1_cost.png', dpi=300)

    plot_optimized(current_state_trajectory, current_input_trajectory, initial_trajectory, U_0)

    return 1

if __name__ == "__main__":
    success = main()
    print("terminated with exit code: ", success)