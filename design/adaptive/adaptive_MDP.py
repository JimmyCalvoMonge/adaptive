from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
import math
from datetime import datetime
import os 
import logging

# Import Markov Decision Module
import MDP

class Adaptive():
    
    def __init__(self, mu, gamma, beta, phi, bs, bi, bz, as1, ai, az, gamma1, tau, delta, t_max, steps, x00):

        ### Logs:
        # A route in my system for logs:
        logger_route = f"C:/Users/jimmy/OneDrive/Desktop/Maestria Metodos Matematicos y Aplicaciones/Tesis/adaptive/design/logs"
        right_now = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
        if not os.path.exists(logger_route):
            os.makedirs(logger_route, exist_ok=True)
        logging.basicConfig(filename=f'{logger_route}/logger_{right_now}_MDP.log', filemode='w', format='%(asctime)s %(message)s',)
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        self.logger = logger

        ### Initial parameters
        self.mu = mu
        self.gamma = gamma
        self.beta = beta
        self.phi = phi
        
        ### R values:
        self.Rmu = self.mu/(self.mu+ self.gamma)
        self.Rphi = self.phi/(self.mu + self.gamma)

        ### Adaptive parameters
        self.bi = bi
        self.bz = bz
        self.bs = bs
        self.ai = ai
        self.az = az
        self.as1 = as1
        self.gamma1 = gamma1
        self.tau = tau
        self.delta = delta

        ### Simulation parameters
        self.t_max = t_max
        self.steps = steps # How many observations between t and t+1
        self.x00 = x00
        self.N = sum(x00)

    def state_odes_system(self, x, t, cs, ci, cz):

        s = x[0]
        i = x[1]
        z = x[2]

        # C function 
        C = cs*ci*self.N/(s*cs + i*ci + z*cz)

        # System 
        dsdt = -C*self.beta*s*(i/self.N) + self.mu*self.N - self.mu*s
        didt = C*self.beta*s*(i/self.N) + self.phi*z*(i/self.N) - (self.gamma + self.mu)*i  
        dzdt = self.gamma*i - self.phi*z*(i/self.N) - self.mu*z

        return [dsdt, didt, dzdt]

    def solve_odes_system_projection(self, x0, t0, cs, ci, cz):

        """
        Solve the classical system with initial conditions
        """

        t = np.linspace(t0, t0 + self.tau, self.steps*self.tau + 1)
        x = odeint(self.state_odes_system, x0, t, args=(cs, ci, cz))

        s = x[:,0]
        i = x[:,1]
        z = x[:,2]

        return s, i, z

    def solve_odes_system_unistep(self, x0, t0, cs, ci, cz):
        
        t = np.linspace(t0, t0 + 1, self.steps + 1)
        x = odeint(self.state_odes_system, x0, t, args=(cs, ci, cz))

        s = x[:,0]
        i = x[:,1]
        z = x[:,2]

        return s, i, z

    def find_optimal_Cs_at_time(self, xt0, ci, cz):
    
        """
        Find the value of C^s at time t.
        x_t0 = [s(t), i(t), z(t)]
        Step 1. For each option for C^s, solve the system in [t,t+tau].
        Step 2. Use the system values to solve the dynamic programming problem for V_t(S).
        Step 3. Find C^s that maximizes V_t(S).
        """

        states = [0, 1, 2]
        actions = np.linspace(0, 0.5*self.bs, 100)
        horizon = self.tau
        delta = self.delta

        S = xt0[0]
        I = xt0[1]
        Z = xt0[2]
        
        phi_t = S*0.5*self.bs + I*ci + Z*cz

        # Immediate rewards:
        def u_s(a):
            return (self.bs*a - a**2)**self.gamma1 - self.as1
        def u_i(a):
            return (self.bi*a - a**2)**self.gamma1 - self.ai
        def u_z(a):
            return (self.bz*a - a**2)**self.gamma1 - self.az

        # Transition Probabilities:
        def P_si(a):
            P_it = 1 - math.exp(-1*(0.5*self.beta*self.bi*I*a)/phi_t)
            return P_it

        def P_ss(a):
            return 1 - P_si(a)
        def P_sz(a):
            return 0

        def P_is(a):
            return 0
        def P_ii(a):
            return math.exp(-1*self.gamma)
        def P_iz(a):
            return 1 - math.exp(-1*self.gamma)

        # No reinfection: (Important) #
        def P_zs(a):
            return 0
        def P_zi(a):
            return 0
        def P_zz(a):
            return 1

        trans_prob_mat = np.array([
            [P_ss, P_si, P_sz],
            [P_is, P_ii, P_iz],
            [P_zs, P_zi, P_zz]
        ])

        reward_vector = np.array([u_s, u_i, u_z])
        trans_probs = [trans_prob_mat]*horizon
        rewards = [reward_vector]*horizon

        # Initialization Point:
        # (0,0,0) doesn't work as an initialization point.
        def vs1(C_st, vti):
            expr0 = 0.5*self.beta*self.bi*I*math.exp(-1*(0.5*self.beta*self.bi*I*C_st)/phi_t)/phi_t
            expr1 = ((self.gamma1*(self.bs*C_st - C_st*2)**(self.gamma1 - 1))*(self.bs - 2*C_st)) / expr0
            expr2 = (1 - P_si(C_st))*expr1 + P_si(C_st)*vti
            return (self.bs*C_st - C_st*2)*self.gamma1 - self.as1 - self.delta*expr2

        C_st_array = np.linspace(0, 0.5*self.bs, 100)
        C_st_tau_step = [vs1(C_st, vti = 0) for C_st in C_st_array]
        Vs1s = max(C_st_tau_step)

        # Use a Markov Decision Process with finite horizon to obtain the optimal policy and decision.
        MDP_adaptive = MDP.MDP(states, actions, rewards, trans_probs, horizon, delta, logger = self.logger)
        MDP_adaptive.fit_optimal_values(verbose = True, init_point = [Vs1s,0,0])
        cs_selected = MDP_adaptive.policies[0][0]

        return cs_selected

    def patch_uni_solutions(self):

        # In each interval [t,t+1] solve the system adaptively.
        # solve_odes_system at t get values at the end, use those for the next one.

        cz = 0.5*self.bz
        ci = 0.5*self.bi

        first_cs = self.find_optimal_Cs_at_time(self.x00, ci, cz)
        s_start, i_start, z_start = self.solve_odes_system_unistep(self.x00, 0, first_cs, ci, cz)

        S = s_start
        I = i_start
        Z = z_start
        Cs = [first_cs]

        for t in range(1, self.t_max):

            # State at end of last interval
            xt_start = [S[-1], I[-1], Z[-1]]
            cs_opt_interval = self.find_optimal_Cs_at_time(xt_start, ci, cz)
            s_interval, i_interval, z_interval = self.solve_odes_system_unistep(xt_start, t, cs_opt_interval, ci, cz)
            S = np.concatenate((S, s_interval), axis=0)
            I = np.concatenate((I, i_interval), axis=0)
            Z = np.concatenate((Z, z_interval), axis=0)
            Cs.append(cs_opt_interval)

        self.S = S
        self.I = I
        self.Z = Z
        self.Cs = Cs

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

    def bifurcation_plot_adaptive(self):
        # TODO
        return None

    def bifurcation_plot_theoretical(self):
        # TODO
        return None