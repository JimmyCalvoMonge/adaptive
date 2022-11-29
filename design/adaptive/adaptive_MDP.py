from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
import math

# Import Markov Decision Module
import MDP

class Adaptive():
    
    def __init__(self, mu, gamma, beta, phi, bs, bi, bz, as1, ai, az, gamma1, tau, delta, t_max, steps, x00):

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

    def state_odes_system(self, x, t, cs, ci, cz):

        s = x[0]
        i = x[1]
        z = x[2]

        # C function 
        C = cs*ci/(s*cs + i*ci + z*cz)

        # System 
        dsdt = -C*self.beta*s*i + self.mu - self.mu*s
        didt = C*self.beta*s*i + self.phi*z*i - (self.gamma + self.mu)*i  
        dzdt = self.gamma*i - self.phi*z*i - self.mu*z

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

    def find_optimal_Cs_at_time(self, t0, xt0, cz, ci):
    
        """
        Find the value of C^s at time t.
        x_t0 = [s(t), i(t), z(t)]

        Step 1. For each option for C^s, solve the system in [t,t+tau].
        Step 2. Use the system values to solve the dynamic programming problem for V_t(S).
        Step 3. Find C^s that maximizes V_t(S).
        """

        states = [0, 1, 2]
        actions = np.linspace(0, 5, 50)
        horizon = self.tau
        delta = self.delta

        # Immediate rewards:
        def u_s(a):
            return (self.bs*a - a**2)**self.gamma1 - self.as1
        def u_i(a):
            return (self.bi*a - a**2)**self.gamma1 - self.ai
        def u_z(a):
            return (self.bz*a - a**2)**self.gamma1 - self.az

        # Transition Probabilities:
        def P_si(a):
            phi_t = xt0[0]*a + xt0[1]*ci + xt0[2]*cz
            probs = 1 - math.exp(-1*(self.beta*xt0[1]*ci*a)/phi_t)
            return probs

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

        # No reinfection:
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

        MDP_adaptive = MDP.MDP(states, actions, rewards, trans_probs, horizon, delta)
        MDP_adaptive.fit_optimal_values(verbose = False)
        cs_selected = MDP_adaptive.actions[0][0]

        # This isn't working :/

        return cs_selected

    def patch_uni_solutions(self):

        # In each interval [t,t+1] solve the system adaptively.
        # solve_odes_system at t get values at the end, use those for the next one.

        cz = 0.5*self.bz
        ci = 0.5*self.bi

        first_cs = self.find_optimal_Cs_at_time(0, self.x00, cz, ci)
        s_start, i_start, z_start = self.solve_odes_system_unistep(self.x00, 0, first_cs, ci, cz)

        S = s_start
        I = i_start
        Z = z_start
        Cs = [first_cs]

        for t in range(1, self.t_max):

            # State at end of last interval
            xt_start = [S[-1], I[-1], Z[-1]]
            cs_opt_interval = self.find_optimal_Cs_at_time(t, xt_start, cz, ci)
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
