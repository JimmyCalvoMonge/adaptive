from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import logging
from tqdm import tqdm


class NonAdaptive():

    def __init__(self, mu, gamma, beta, phi, Cs, Ci, Cz, x00, t_max, **kwargs):

        # Logs:

        logger_route = "C:/Users/jimmy/OneDrive/Desktop/Maestria Metodos Matematicos y Aplicaciones/Tesis/adaptive/code/logs"
        right_now = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
        if not os.path.exists(logger_route):
            os.makedirs(logger_route, exist_ok=True)
        logging.basicConfig(filename=f'{logger_route}/logger_{right_now}.log',
                            filemode='w', format='%(asctime)s %(message)s',)
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        self.logger = logger

        # Initial parameters
        self.mu = mu
        self.gamma = gamma
        self.beta = beta
        self.phi = phi

        # R values:
        self.Rmu = self.mu/(self.mu + self.gamma)
        self.Rphi = self.phi/(self.mu + self.gamma)

        # Adaptive parameters
        self.Cs = Cs
        self.Ci = Ci
        self.Cz = Cz

        # Simulation parameters
        self.t_max = t_max
        self.x00 = x00
        self.N = sum(x00)

        # For odeint
        self.steps = kwargs.get('steps', 1000)

        # For RK4
        self.dt = kwargs.get('dt', 0.01)

        # tqdm
        self.tqdm = kwargs.get('tqdm', True)

    # ============ Using python's odeint method =========== #
    
    def state_odes_system(self, x, t):

        s = x[0]
        i = x[1]
        z = x[2]

        # C function
        C = self.Cs*self.Ci*self.N/(s*self.Cs + i*self.Ci + z*self.Cz)

        # System
        dsdt = -C*self.beta*s*(i/self.N) + self.mu*self.N - self.mu*s
        didt = C*self.beta*s*(i/self.N) + self.phi*z * \
            (i/self.N) - (self.gamma + self.mu)*i
        dzdt = self.gamma*i - self.phi*z*(i/self.N) - self.mu*z

        return [dsdt, didt, dzdt]

    def solve_odes_system_odeint(self):
        """
        Solve the classical system with initial conditions
        """

        t = np.linspace(0, 0 + self.t_max, self.steps)
        self.time = t
        x = odeint(func=self.state_odes_system,
                   y0=self.x00, t=t, full_output=True)

        self.S = x[0][:, 0]
        self.I = x[0][:, 1]
        self.Z = x[0][:, 2]

        self.full_solution = x

    # ============ Using Four Step Runge-Kutta ============ #
    # Taken from https://medium.com/geekculture/runge-kutta-numerical-integration-of-ordinary-differential-equations-in-python-9c8ab7fb279c

    def ode_system(self, _t, _y):
        """
        system of first order differential equations
        _t: discrete time step value
        _y: state vector [y1, y2, y3]
        """

        C = self.Cs*self.Ci*self.N / \
            (_y[0]*self.Cs + _y[1]*self.Ci + _y[2]*self.Cz)

        dsdt = -C*self.beta*_y[0]*(_y[1]/self.N) + \
            self.mu*self.N - self.mu*_y[0]
        didt = C*self.beta*_y[0]*(_y[1]/self.N) + self.phi * \
            _y[2]*(_y[1]/self.N) - (self.gamma + self.mu)*_y[1]
        dzdt = self.gamma*_y[1] - self.phi*_y[2]*(_y[1]/self.N) - self.mu*_y[2]

        return np.array([dsdt, didt, dzdt])

    def rk4(self, func, tk, _yk, _dt=0.01, **kwargs):
        """
        single-step fourth-order numerical integration (RK4) method
        func: system of first order ODEs
        tk: current time step
        _yk: current state vector [y1, y2, y3, ...]
        _dt: discrete time step size
        **kwargs: additional parameters for ODE system
        returns: y evaluated at time k+1
        """

        # evaluate derivative at several stages within time interval
        f1 = func(tk, _yk, **kwargs)
        f2 = func(tk + _dt / 2, _yk + (f1 * (_dt / 2)), **kwargs)
        f3 = func(tk + _dt / 2, _yk + (f2 * (_dt / 2)), **kwargs)
        f4 = func(tk + _dt, _yk + (f3 * _dt), **kwargs)

        # return an average of the derivative over tk, tk + dt
        return _yk + (_dt / 6) * (f1 + (2 * f2) + (2 * f3) + f4)

    def solve_odes_system_RK4(self):

        dt = self.dt
        time = np.arange(0.0, self.t_max + dt, dt)
        self.time = time

        # second order system initial conditions [y1, y2] at t = 1
        y0 = np.array(self.x00)

        # ==============================================================
        # propagate state

        # simulation results
        state_history = []

        # initialize yk
        yk = y0

        # intialize time
        t = 0

        range_use = range(time) 
        if self.tqdm:
            range_use = tqdm(time)

        # approximate y at time t
        for t in tqdm(time):
            state_history.append(yk)
            yk = self.rk4(self.ode_system, t, yk, dt)

        # convert list to numpy array
        state_history = np.array(state_history)

        self.full_solution = state_history

        self.S = state_history[:, 0]
        self.I = state_history[:, 1]
        self.Z = state_history[:, 2]

    def solve_odes_system(self, **kwargs):

        method = kwargs.get('method', 'RK4')

        # print(f" Started solving system of ODEs. Method: {method}. ")

        if method == 'RK4':
            self.solve_odes_system_RK4()
        else:
            self.solve_odes_system_odeint()

    def plot_ode_solution(self, **kwargs):

        title = kwargs.get('title','Plot')

        plt.plot(self.time, self.S, label="Susceptible")
        plt.plot(self.time, self.I, label="Infected")
        plt.plot(self.time, self.Z, label="Recovered")
        plt.title(f"Plot of S-I-Z functions ({title})")
        plt.xlabel("Time (t)")
        plt.ylabel("Number of individuals")
        plt.legend(loc="upper right")
        plt.rcParams["figure.figsize"] = (10, 6)
        plt.show()
