from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
import math
from datetime import datetime
import os
import logging
from tqdm import tqdm
import sys
# Import Markov Decision Module
base_path = r'C:\Users\jimmy\OneDrive\Desktop\Maestria Metodos Matematicos y Aplicaciones\Tesis\git\code'
sys.path.insert(0, f'{base_path}/adaptive/source_code')
sys.path.insert(0, f'{base_path}/disaggregated/source_code')
MDP = __import__('MDP', globals=None, locals=None, fromlist=(), level=0)

# ========== Adaptive Method using utility
# functions as input parameters ========== #
# Using S-E-I-R model


class Adaptive_SEIR():

    def __init__(self,
                 mu, gamma, beta, phi, kappa,
                 tau, delta,
                 u_s, u_e, u_i, u_z,
                 t_max, steps, x00,
                 **kwargs):

        # Logs:
        logs = kwargs.get('logs', False)
        if logs:
            # A route in my system for logs:
            logger_route = f"{base_path}/logs"
            right_now = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

            if not os.path.exists(logger_route):
                os.makedirs(logger_route, exist_ok=True)

            logging.basicConfig(
                filename=f'{logger_route}/logger_{right_now}_MDP.log',
                filemode='w', format='%(asctime)s %(message)s',)
            logger = logging.getLogger()
            logger.setLevel(logging.INFO)
            self.logger = logger
        else:
            self.logger = False

        # Initial parameters
        self.mu = mu
        self.gamma = gamma
        self.beta = beta
        self.phi = phi
        self.kappa = kappa

        # R values:
        self.Rmu = self.mu/(self.mu + self.gamma)
        self.Rphi = self.phi/(self.mu + self.gamma)

        # Adaptive parameters
        self.tau = tau
        self.delta = delta

        # Utility functions
        self.u_s = u_s
        self.u_e = u_e
        self.u_i = u_i
        self.u_z = u_z

        # Simulation parameters
        self.t_max = t_max
        self.steps = steps  # How many observations between t and t+1
        self.x00 = x00
        self.N = sum(x00)
        max_contacts = kwargs.get('max_contacts', None)
        if max_contacts:
            self.actions = np.linspace(0, max_contacts, 100)
        else:
            self.actions = kwargs.get('actions', np.linspace(0, 30, 100))

        # Global maxima of utility functions
        self.cs_max = self.actions[np.nanargmax(
            [self.u_s(a) for a in self.actions])]
        self.ce_max = self.actions[np.nanargmax(
            [self.u_e(a) for a in self.actions])]
        self.ci_max = self.actions[np.nanargmax(
            [self.u_i(a) for a in self.actions])]
        if self.ci_max == 0:
            self.ci_max = self.cs_max
        self.cz_max = self.actions[np.nanargmax(
            [self.u_z(a) for a in self.actions])]

        self.inf_prob_mult = kwargs.get('inf_prob_mult', 0.7)
        self.init_point = kwargs.get('init_point', None)
        self.verbose = kwargs.get('verbose', False)
        self.tqdm = kwargs.get('tqdm', True)

        # Compute solution until
        # (S_t-S_{t+1})^2 + (I_t-I_{t+1})^2 + (I_t-I_{t+1})^2 < threshold
        self.compute_max_t_threshold = kwargs.get(
            'compute_max_t_threshold', None)
        self.stopping_point = 0

    def state_odes_system(self, x, t, cs, ce, ci, cz):

        s = x[0]
        e = x[1]
        i = x[2]
        z = x[3]

        # g function
        incidence_rate = cs*ci*self.N/(s*cs + e*ce + i*ci + z*cz)

        # System S-E-I-R
        dsdt = -incidence_rate*self.beta*s*(i/self.N) + self.mu*self.N - self.mu*s
        dedt = incidence_rate*self.beta*s*(i/self.N) - (self.kappa + self.mu)*e
        didt = self.kappa*e + self.phi*z*(i/self.N) - (self.gamma + self.mu)*i
        dzdt = self.gamma*i - self.phi*z*(i/self.N) - self.mu*z

        return [dsdt, dedt, didt, dzdt]

    def solve_odes_system_projection(self, x0, t0, cs, ce, ci, cz):
        """
        Solve the classical system with initial conditions
        """

        t = np.linspace(t0, t0 + self.tau, self.steps*self.tau + 1)
        x = odeint(self.state_odes_system, x0, t, args=(cs, ce, ci, cz))

        s = x[:, 0]
        e = x[:, 1]
        i = x[:, 2]
        z = x[:, 3]

        return s, e, i, z

    def solve_odes_system_unistep(self, x0, t0, cs, ce, ci, cz):

        t = np.linspace(t0, t0 + 1, self.steps + 1)
        x = odeint(self.state_odes_system, x0, t, args=(cs, ce, ci, cz))

        s = x[:, 0]
        e = x[:, 1]
        i = x[:, 2]
        z = x[:, 3]

        return s, e, i, z

    def find_optimal_C_at_time(self, xt0, cs_star, ce_star, ci_star, cz_star):
        """
        Find the value of C^s at time t.
        x_t0 = [s(t), i(t), z(t)]
        Step 1. For each option for C^s, solve the system in [t,t+tau].
        Step 2. Use the system values to solve
        the dynamic programming problem for V_t(S).
        Step 3. Find C^s that maximizes V_t(S).
        """

        states = [0, 1, 2, 3]
        actions = self.actions
        horizon = self.tau
        delta = self.delta

        s_proj, e_proj, i_proj, z_proj = self.solve_odes_system_projection(
            x0=xt0, t0=0,
            cs=cs_star, ce=ce_star, ci=ci_star, cz=cz_star)

        trans_probs = []
        for tt in range(horizon):

            # Transition Probabilities (at time t):
            # Using the system projection

            # S -> _
            def P_se(a):
                phi_t = s_proj[tt*self.steps]*cs_star + \
                    e_proj[tt*self.steps]*ce_star + \
                    i_proj[tt*self.steps]*ci_star + \
                    z_proj[tt*self.steps]*cz_star

                P_it = 1 - math.exp(-1*(self.beta*ci_star*i_proj[tt*self.steps]*a)/phi_t)
                return P_it

            def P_ss(a):
                return 1 - P_se(a)
            
            def P_si(a):
                return 0

            def P_sz(a):
                return 0

            # E -> _
            def P_es(a):
                return 0
            
            def P_ee(a):
                return math.exp(-1*self.kappa)
            
            def P_ei(a):
                return 1 - P_ee(a)
            
            def P_ez(a):
                return 0

            # I -> _
            def P_is(a):
                return 0
            
            def P_ie(a):
                return 0
            
            def P_ii(a):
                return math.exp(-1*self.gamma)

            def P_iz(a):
                return 1 - math.exp(-1*self.gamma)

            # Z -> _
            def P_zs(a):
                return 0
            
            def P_ze(a):
                return 0

            def P_zi(a):
                if self.phi == 0:
                    return 0
                return P_se(a)*0.97

            def P_zz(a):
                return 1 - P_zi(a)

            trans_prob_mat = np.array([
                [P_ss, P_se, P_si, P_sz],
                [P_es, P_ee, P_ei, P_ez],
                [P_is, P_ie, P_ii, P_iz],
                [P_zs, P_ze, P_zi, P_zz]
            ])
            trans_probs.append(trans_prob_mat)

        reward_vector = np.array([self.u_s, self.u_e, self.u_i, self.u_z])
        rewards = [reward_vector]*horizon

        """
        Initialization point for MDP process
        """

        if self.init_point:
            init_point_use = self.init_point
        else:
            init_point_use = [np.nanmax([self.u_s(a) for a in actions]),
                              np.nanmax([self.u_e(a) for a in actions]),
                              0,
                              np.nanmax([self.u_z(a) for a in actions])]

        # Use a Markov Decision Process with
        # finite horizon to obtain the optimal policy and decision.
        MDP_adaptive = MDP.MDP(states, actions, rewards,
                               trans_probs, horizon, delta,
                               logger=self.logger, verbose=self.verbose)
        MDP_adaptive.fit_optimal_values(init_point=init_point_use)
        return MDP_adaptive.policies, MDP_adaptive.values_history

    def patch_uni_solutions(self):

        # In each interval [t,t+1] solve the system adaptively.
        # solve_odes_system at t get values at the end,
        # use those for the next one.
        S, E, I, Z = [], [], [], []
        cs_history, ce_history, ci_history, cz_history = [], [], [], []
        val_func_vals = []
        # print("Patching unit time solutions ...")

        def compute_uni_solution(t):
            # State at end of last interval
            if t == 0:
                xt_start = [self.x00[0], self.x00[1], self.x00[2], self.x00[3]]
                cs_use, ce_use, ci_use, cz_use = self.cs_max, self.ce_max, self.ci_max, self.cz_max
            else:
                cs_use, ce_use, ci_use, cz_use = cs_history[-1], ce_history[-1], \
                    ci_history[-1], cz_history[-1]
                xt_start = [S[-1], E[-1], I[-1], Z[-1]]

            cs_policies, \
                cs_values = self.find_optimal_C_at_time(
                    xt_start, cs_use, ce_use, ci_use, cz_use)
            val_func_vals.append(cs_values)

            s_interval, e_interval, i_interval, \
                z_interval = self.solve_odes_system_unistep(
                    xt_start, t,
                    cs_policies[0][0],
                    cs_policies[0][1],
                    cs_policies[0][2],
                    cs_policies[0][3]
                    )

            return s_interval, e_interval, i_interval, z_interval, cs_policies

        # Print tqdm or not.
        if self.tqdm:
            range_use = tqdm(range(0, self.t_max))
        else:
            range_use = range(0, self.t_max)

        for t in range_use:

            s_interval, e_interval, i_interval, \
                z_interval, cs_policies = compute_uni_solution(t)

            if self.compute_max_t_threshold and t > 1000:

                diff_vect = np.array(
                    [S[-1] - s_interval[-1],
                     E[-1] - e_interval[-1],
                     I[-1] - i_interval[-1],
                     Z[-1] - z_interval[-1]])
                diff_vect = (np.linalg.norm(diff_vect, ord=2))**2

                if diff_vect < self.compute_max_t_threshold:
                    self.stopping_point = t
                    # print(f"Stopped, found stopping condition at {t}")
                    break

            S = np.concatenate((S, s_interval), axis=0)
            E = np.concatenate((E, e_interval), axis=0)
            I = np.concatenate((I, i_interval), axis=0)
            Z = np.concatenate((Z, z_interval), axis=0)

            cs_history.append(cs_policies[0][0])
            ce_history.append(cs_policies[0][1])
            ci_history.append(cs_policies[0][2])
            cz_history.append(cs_policies[0][3])

        self.S = S
        self.E = E
        self.I = I
        self.Z = Z

        self.cs_history = cs_history
        self.ce_history = ce_history
        self.ci_history = ci_history
        self.cz_history = cz_history

        self.val_func_vals = val_func_vals

        # END

    def plot_ode_solution(self, t, title):

        plt.plot(t, self.S, label="Susceptible", color='blue', linewidth=3)
        plt.plot(t, self.E, label="Exposed", color='orange', linewidth=3)
        plt.plot(t, self.I, label="Infected", color='red', linewidth=3)
        plt.plot(t, self.Z, label="Recovered", color='limegreen', linewidth=3)
        plt.title(f"Plot of S-E-I-R functions ({title})")
        plt.xlabel("Time (t)")
        plt.ylabel("Number of individuals")
        plt.legend(loc="upper right")
        plt.rcParams["figure.figsize"] = (10, 6)
        plt.show()

    def plot_C(self, Cs, **kwargs):
        color = kwargs.get('color', 'blue')
        cs_unistep = []
        for i in tqdm(range(len(Cs))):
            cs_unistep = cs_unistep + [Cs[i]]*self.steps

        plt.plot(np.linspace(0, self.t_max, len(cs_unistep)),
                 cs_unistep, label="Contacts", linewidth=3, color=color)
        plt.xlabel("Time (t)")
        plt.ylabel("Optimal contact selected")
        plt.legend(loc="upper right")
        plt.rcParams["figure.figsize"] = (10, 6)
        plt.show()