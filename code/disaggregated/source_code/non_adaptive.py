from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
import os

# base_path
from inspect import currentframe, getframeinfo
from pathlib import Path
filename = getframeinfo(currentframe()).filename
# ./code/adaptive/source_code
file_path_parent = Path(filename).resolve().parent
base_path = os.path.dirname(os.path.dirname(file_path_parent))  # ./code


class NonAdaptive():

    def __init__(self, mu, gamma, beta, phi, Cs, Ci, Cz, x00, t_max, **kwargs):

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

    def solve_odes_system(self):
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

    def plot_ode_solution(self, **kwargs):

        title = kwargs.get('title', 'Plot')

        plt.plot(self.time, self.S, label="Susceptible")
        plt.plot(self.time, self.I, label="Infected")
        plt.plot(self.time, self.Z, label="Recovered")
        plt.title(f"Plot of S-I-Z functions ({title})")
        plt.xlabel("Time (t)")
        plt.ylabel("Number of individuals")
        plt.legend(loc="upper right")
        plt.rcParams["figure.figsize"] = (10, 6)
        plt.show()
