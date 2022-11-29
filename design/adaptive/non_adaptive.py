from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
import math
from datetime import datetime
import os 
import logging

class NonAdaptive():
    
    def __init__(self, mu, gamma, beta, phi, Cs, Ci, Cz, t_max, steps, x00):

        ### Logs:

        logger_route=f"C:/Users/jimmy/OneDrive/Desktop/Maestria Metodos Matematicos y Aplicaciones/Tesis/adaptive/design/logs"
        right_now = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
        if not os.path.exists(logger_route):
            os.makedirs(logger_route, exist_ok=True)
        logging.basicConfig(filename=f'{logger_route}/logger_{right_now}.log',filemode='w', format='%(asctime)s %(message)s',)
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        self.logger = logger

        ### Initial parameters
        self.mu = mu
        self.gamma = gamma
        self.beta = beta
        self.phi = phi
        
        ### R values:
        # self.Rmu = self.mu/(self.mu+ self.gamma)
        # self.Rphi = self.phi/(self.mu + self.gamma)

        ### Adaptive parameters
        self.Cs = Cs
        self.Ci = Ci
        self.Cz = Cz

        ### Simulation parameters
        self.t_max = t_max
        self.steps = steps # How many observations between t and t+1
        self.x00 = x00

    def state_odes_system(self, x, t):

        s = x[0]
        i = x[1]
        z = x[2]

        # C function 
        C = self.Cs*self.Ci/(s*self.Cs + i*self.Ci + z*self.Cz)

        # System 
        dsdt = -C*self.beta*s*i + self.mu - self.mu*s
        didt = C*self.beta*s*i + self.phi*z*i - (self.gamma + self.mu)*i  
        dzdt = self.gamma*i - self.phi*z*i - self.mu*z

        return [dsdt, didt, dzdt]

    def solve_odes_system(self):

        """
        Solve the classical system with initial conditions
        """

        t = np.linspace(0, 0 + self.t_max, self.steps)
        x = odeint(self.state_odes_system, self.x00, t)

        s = x[:,0]
        i = x[:,1]
        z = x[:,2]

        self.S = s
        self.I = i
        self.Z = z

    def plot_ode_solution(self, t, title):

        plt.plot(t, self.S, label="Susceptible")
        plt.plot(t, self.I, label="Infected")
        plt.plot(t, self.Z, label="Recovered")
        plt.title(f"Plot of S-I-Z functions ({title})")
        plt.xlabel("Time (t)")
        plt.ylabel("Number of individuals")
        plt.legend(loc = "upper right")
        plt.rcParams["figure.figsize"] = (10,6)
        plt.show()
