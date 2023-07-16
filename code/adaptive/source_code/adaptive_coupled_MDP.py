"""
Coupled Adaptive Class
For Vector-Borne Diseases system:



"""
from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
import math
from datetime import datetime
import os
import logging
from tqdm import tqdm

# Import Markov Decision Module
import MDP

# base_path
from inspect import currentframe, getframeinfo
from pathlib import Path
filename = getframeinfo(currentframe()).filename
# ./code/adaptive/source_code
file_path_parent = Path(filename).resolve().parent
base_path = os.path.dirname(os.path.dirname(file_path_parent))  # ./code

class Adaptive_Coupled():

    def __init__(self,
                 mu_1, mu_2,
                 alpha_1, alpha_2,
                 beta_1, beta_2,
                 gamma, phi,
                 tau, delta,
                 u_s1, u_e1, u_i1, u_z1,
                 cs_2, ce_2, ci_2,
                 t_max, steps,
                 x00_1, x00_2,
                 max_contacts_1,
                 max_contacts_2,
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
        self.mu_1 = mu_1
        self.mu_2 = mu_2
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.gamma = gamma
        self.phi = phi

        # Adaptive parameters
        self.tau = tau
        self.delta = delta

        # Utility functions

        # 1st Agent type
        self.u_s1 = u_s1
        self.u_i1 = u_i1
        self.u_e1 = u_e1
        self.u_z1 = u_z1

        # 2nd Agent type
        self.cs_2 = cs_2
        self.ce_2 = ce_2
        self.ci_2 = ci_2

        # Simulation parameters
        self.t_max = t_max
        self.steps = steps  # How many observations between t and t  +1

        self.x00_1 = x00_1
        self.N_1 = sum(x00_1)
        self.x00_2 = x00_2
        self.N_2 = sum(x00_2)

        self.max_contacts_1 = max_contacts_1
        self.max_contacts_2 = max_contacts_2

        self.actions_1 = np.linspace(0, self.max_contacts_1, 100)
        self.actions_2 = np.linspace(0, self.max_contacts_2, 100)

        # Utility Optimal Points for Hosts#
        ut_maxs_1 = []
        
        for utFn in [self.u_s1, self.u_e1, self.u_i1, self.u_z1]:
            ut_maxs_1.append(self.actions_1[np.nanargmax(
            [utFn(a) for a in self.actions_1])])


        if ut_maxs_1[2] == 0:
            ut_maxs_1[1] = ut_maxs_1[0]

        self.cs_max_1 = ut_maxs_1[0]
        self.ce_max_1 = ut_maxs_1[1]
        self.ci_max_1 = ut_maxs_1[2]
        self.cz_max_1 = ut_maxs_1[3]

        self.inf_prob_mult = kwargs.get('inf_prob_mult', 0.7)
        self.init_point = kwargs.get('init_point', None)
        self.verbose = kwargs.get('verbose', False)
        self.tqdm = kwargs.get('tqdm', True)

        # Compute solution until
        # ||X_t - X_{t+1}|| < threshold
        self.compute_max_t_threshold = kwargs.get(
            'compute_max_t_threshold', None)
        self.stopping_point = 0

    def state_odes_system(self, x, t, cs_1, ce_1, ci_1, cz_1, cs_2, ce_2, ci_2):

        s_1 = x[0]
        e_1 = x[1]
        i_1 = x[2]
        z_1 = x[3]

        s_2 = x[4]
        e_2 = x[5]
        i_2 = x[6]

        denom = s_1*cs_1 + e_1*ce_1 + i_1*ci_1 + z_1*cz_1 + s_2*cs_2 + e_2*ce_2 + i_2*ci_2

        # C values:
        C_1 = (cs_1*ci_2*i_2)/(denom)
        C_2 = (cs_2*ci_1*i_1)/(denom)

        # System

        # Hosts
        ds_1dt = self.mu_1*self.N_1 - self.beta_1*C_1*s_1*(i_2/self.N_2) - self.mu_1*s_1
        de_1dt = self.beta_1*C_1*s_1*(i_2/self.N_2) - (self.mu_1 + self.alpha_1)*e_1
        di_1dt = self.alpha_1*e_1 - (self.mu_1 + self.gamma)*i_1
        dz_1dt = self.gamma*i_1 - self.mu_1*z_1

        # Vectors
        ds_2dt = self.mu_2*self.N_2 - self.beta_2*C_2*s_2*(i_1/self.N_1) - self.mu_2*s_2
        de_2dt = self.beta_1*C_2*s_2*(i_1/self.N_1) - (self.mu_2 + self.alpha_2)*e_2
        di_2dt = self.alpha_2*e_2 - self.mu_2*i_2

        return [ds_1dt, de_1dt, di_1dt, dz_1dt, ds_2dt, de_2dt, di_2dt]

    def solve_odes_system_projection(self, x0, t0, cs_1, ce_1, ci_1, cz_1, cs_2, ce_2, ci_2):

        """
        Solve the classical system with initial conditions
        """

        t = np.linspace(t0, t0 + self.tau, self.steps*self.tau + 1)
        x = odeint(self.state_odes_system, x0, t, args=(cs_1, ce_1, ci_1, cz_1, cs_2, ce_2, ci_2))

        s_1 = x[:, 0]
        e_1 = x[:, 1]
        i_1 = x[:, 2]
        z_1 = x[:, 3]

        s_2 = x[:, 4]
        e_2 = x[:, 5]
        i_2 = x[:, 6]

        return s_1, e_1, i_1, z_1, s_2, e_2, i_2

    def solve_odes_system_unistep(self, x0, t0, cs_1, ce_1, ci_1, cz_1, cs_2, ce_2, ci_2):

        t = np.linspace(t0, t0 + 1, self.steps + 1)
        x = odeint(self.state_odes_system, x0, t, args=(cs_1, ce_1, ci_1, cz_1, cs_2, ce_2, ci_2))

        s_1 = x[:, 0]
        e_1 = x[:, 1]
        i_1 = x[:, 2]
        z_1 = x[:, 3]

        s_2 = x[:, 4]
        e_2 = x[:, 5]
        i_2 = x[:, 6]

        return s_1, e_1, i_1, z_1, s_2, e_2, i_2

    def find_optimal_C_at_time(self, xt0, C_vector):

        """
        Find the value of C^s at time t.
        x_t0 = [s(t), i(t), z(t)]
        Step 1. For each option for C^s, solve the system in [t,t+tau].
        Step 2. Use the system values to solve
        the dynamic programming problem for V_t(S).
        Step 3. Find C^s that maximizes V_t(S).
        Do this for Hosts Only

        Need to define the following probability transition matrix:

             P_s1s1   P_s1e1    P_s1i1    P_s1z1
        P =  P_e1s1   P_e1e1    P_e1i1    P_e1z1
             P_i1s1   P_i1e1    P_i1i1    P_i1z1
             P_z1s1   P_z1e1    P_z1i1    P_z1z1


             P_s1s1   P_s1e1    0         0
        P =  0        P_e1e1    P_e1i1    0
             0        0         P_i1i1    P_i1z1
             0        P_z1e1    0         P_z1z1
    
        """

        states = [0, 1, 2, 3]
        actions = self.actions_1
        horizon = self.tau
        delta = self.delta

        s1_proj, e1_proj, i1_proj, z1_proj, \
                s2_proj, e2_proj, i2_proj = self.solve_odes_system_projection(
            x0=xt0, t0=0, cs_1=C_vector[0], ce_1=C_vector[1],
            ci_1=C_vector[2], cz_1=C_vector[3],
            cs_2=C_vector[4], ce_2=C_vector[5],
            ci_2=C_vector[6])
        trans_probs = []

        for tt in range(horizon):

            # Transition Probabilities (at time t):
            # Using the system projection

            # Row P_{s.}
            def P_se(a):
                phi_t = s1_proj[tt*self.steps]*C_vector[0] + \
                    e1_proj[tt*self.steps]*C_vector[1] + \
                    i1_proj[tt*self.steps]*C_vector[2] + \
                    z1_proj[tt*self.steps]*C_vector[3] + \
                    s2_proj[tt*self.steps]*C_vector[4] + \
                    e2_proj[tt*self.steps]*C_vector[5] + \
                    i2_proj[tt*self.steps]*C_vector[6]
                P_et = 1 - math.exp(-1*(self.beta_1*C_vector[0]*
                                    i2_proj[tt*self.steps]*a)/phi_t)
                return P_et

            def P_ss(a):
                return 1 - P_se(a)
            
            def P_si(a):
                return 0

            def P_sz(a):
                return 0

            # Row P_{e.}
            def P_es(a):
                return 0
            
            def P_ee(a):
                return 0
            
            def P_ei(a):
                return 1

            def P_ez(a):
                return 0

            # Row P_{i.}
            def P_is(a):
                return 0

            def P_ie(a):
                return 0

            def P_ii(a):
                return math.exp(-1*self.gamma)

            def P_iz(a):
                return 1 - math.exp(-1*self.gamma)
            
            # Row P_{z.}
            def P_zs(a):
                return 0
        
            def P_ze(a):
                if self.phi == 0:
                    return 0
                return P_se(a)*0.97

            def P_zi(a):
                return 0

            def P_zz(a):
                return 1 - P_ze(a)

            trans_prob_mat = np.array([
                [P_ss, P_se, P_si, P_sz],
                [P_es, P_ee, P_ei, P_ez],
                [P_is, P_ie, P_ii, P_iz],
                [P_zs, P_ze, P_zi, P_zz]
            ])
            trans_probs.append(trans_prob_mat)

        reward_vector = np.array([self.u_s1, self.u_e1, self.u_i1, self.u_z1])
        rewards = [reward_vector]*horizon

        """
        Initialization point for MDP process
        """

        if self.init_point:
            init_point_use = self.init_point
        else:
            init_point_use = [
                np.nanmax([self.u_s1(a) for a in actions]),
                np.nanmax([self.u_e1(a) for a in actions]),
                0,
                np.nanmax([self.u_z1(a) for a in actions])]

        # Use a Markov Decision Process with
        # finite horizon to obtain the optimal policy and decision.
        MDP_adaptive = MDP.MDP(states, actions, rewards,
                            trans_probs, horizon, delta,
                            logger=self.logger,
                            verbose=self.verbose)
        MDP_adaptive.fit_optimal_values(init_point=init_point_use)
        
        return MDP_adaptive.policies, MDP_adaptive.values_history

    def patch_uni_solutions(self):

        # In each interval [t,t+1] solve the system adaptively.
        # solve_odes_system at t get values at the end,
        # use those for the next one.
        S_1, E_1, I_1, Z_1 = [], [], [], []
        S_2, E_2, I_2 = [], [], []

        cs_1_history, ce_1_history, ci_1_history, cz_1_history = [], [], [], []
        val_func_vals = []

        def compute_uni_solution(t):
            # State at end of last interval
            if t == 0:
                xt_start = [self.x00_1[0], self.x00_1[1], self.x00_1[2], self.x00_1[3],
                            self.x00_2[0], self.x00_2[1], self.x00_2[2]
                            ]
                cs_1_use, ce_1_use, ci_1_use, cz_1_use = self.cs_max_1, self.ce_max_1, self.ci_max_1, self.cz_max_1

            else:
                xt_start = [S_1[-1], E_1[-1], I_1[-1], Z_1[-1], S_2[-1], E_2[-1], I_2[-1]]
                cs_1_use, ce_1_use, ci_1_use, cz_1_use = cs_1_history[-1], ce_1_history[-1], \
                    ci_1_history[-1], cz_1_history[-1]

            C_vector = [cs_1_use, ce_1_use, ci_1_use, cz_1_use, self.cs_2, self.ce_2, self.ci_2]
            
            # Optimal contact decision for Hosts
            cs_policies_1, \
                cs_values_1 = self.find_optimal_C_at_time(
                    xt_start, C_vector, type='Host')

            val_func_vals.append(cs_values_1)

            s_1_interval, e_1_interval, i_1_interval, z_1_interval, \
                s_2_interval, e_2_interval, i_2_interval = self.solve_odes_system_unistep(
                    xt_start, t, cs_policies_1[0][0],
                    cs_policies_1[0][1], cs_policies_1[0][2],
                    cs_policies_1[0][3], self.cs_2,
                    self.ce_2, self.ci_2,
                    )

            return s_1_interval, e_1_interval, i_1_interval, z_1_interval, \
                s_2_interval, e_2_interval, i_2_interval, cs_policies_1

        # Print tqdm or not.
        if self.tqdm:
            range_use = tqdm(range(0, self.t_max))
        else:
            range_use = range(0, self.t_max)

        for t in range_use:

            s_1_interval, e_1_interval, i_1_interval, z_1_interval, \
                s_2_interval, e_2_interval, i_2_interval, cs_policies_1 = compute_uni_solution(t)

            if self.compute_max_t_threshold and t > 1000:

                diff_vect = np.array(
                    [
                        S_1[-1] - s_1_interval[-1],
                        E_1[-1] - e_1_interval[-1],
                        I_1[-1] - i_1_interval[-1],
                        Z_1[-1] - z_1_interval[-1],
                        S_2[-1] - s_2_interval[-1],
                        E_2[-1] - e_2_interval[-1],
                        I_2[-1] - i_2_interval[-1]
                    ])
                diff_vect = (np.linalg.norm(diff_vect, ord=2))**2

                if diff_vect < self.compute_max_t_threshold:
                    self.stopping_point = t
                    # print(f"Stopped, found stopping condition at {t}")
                    break

            S_1= np.concatenate((S_1, s_1_interval), axis=0)
            E_1= np.concatenate((E_1, e_1_interval), axis=0)
            I_1= np.concatenate((I_1, i_1_interval), axis=0)
            Z_1= np.concatenate((Z_1, z_1_interval), axis=0)
            S_2= np.concatenate((S_2, s_2_interval), axis=0)
            E_2= np.concatenate((E_2, e_2_interval), axis=0)
            I_2= np.concatenate((I_2, i_2_interval), axis=0)

            cs_1_history.append(cs_policies_1[0][0])
            ce_1_history.append(cs_policies_1[0][1])
            ci_1_history.append(cs_policies_1[0][2])
            cz_1_history.append(cs_policies_1[0][3])

        self.S_1 = S_1
        self.E_1 = E_1
        self.I_1 = I_1
        self.Z_1 = Z_1
        self.S_2 = S_2
        self.E_2 = E_2
        self.I_2 = I_2

        self.cs_1_history = cs_1_history
        self.ce_1_history = ce_1_history
        self.ci_1_history = ci_1_history
        self.cz_1_history = cz_1_history

        self.val_func_vals = val_func_vals

    def plot_ode_solution(self, t, title):

        plt.plot(t, self.S_1, label="Susceptible (Host)")
        plt.plot(t, self.E_1, label="Exposed (Host)")
        plt.plot(t, self.I_1, label="Infected (Host)")
        plt.plot(t, self.Z_1, label="Recovered (Host)")

        plt.plot(t, self.S_2, label="Susceptible (Vector)")
        plt.plot(t, self.E_2, label="Exposed (Vector)")
        plt.plot(t, self.I_2, label="Infected (Vector)")

        plt.title(f"Plot of S-I-Z functions ({title})")
        plt.xlabel("Time (t)")
        plt.ylabel("Number of individuals")
        plt.legend(loc="upper right")
        plt.rcParams["figure.figsize"] = (10, 6)
        plt.show()

    def plot_C(self, Cs):
        cs_unistep = []
        for i in tqdm(range(len(Cs))):
            cs_unistep = cs_unistep + [Cs[i]]*self.steps
        plt.plot(np.linspace(0, self.t_max, len(cs_unistep)),
                 cs_unistep, label="C^s", linewidth=3)
        plt.xlabel("Time (t)")
        plt.ylabel("Optimal contact selected")
        plt.legend(loc="upper right")
        plt.rcParams["figure.figsize"] = (10, 6)
        plt.show()