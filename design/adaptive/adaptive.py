from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
import math
from datetime import datetime
import os 
import logging

class Adaptive():
    
    def __init__(self, mu, gamma, beta, phi, bs, bi, bz, as1, ai, az, gamma1, tau, delta, t_max, steps, x00):

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

    def utility_s(self, cs):
        return (self.bs*cs - cs**2)**self.gamma1 - self.as1

    def prob_i(self, cs, cs0, ci, cz, s, i , z):
        phi_t = s*cs0 + i*ci + z*cz
        P_it = 1 - math.exp(-1*(self.beta*i*ci*cs)/phi_t)
        return P_it

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

    def find_optimal_Cs_at_time(self, t0, xt0, ci, cz):
        
        ### Old Version ###
        """
        Find the value of C^s at time t.
        x_t0 = [s(t), i(t), z(t)]

        Step 1. For each option for C^s, solve the system in [t,t+tau].
        Step 2. Use the system values to solve the dynamic programming problem for V_t(S).
        Step 3. Find C^s that maximizes V_t(S).
        """

        S = xt0[0]
        I = xt0[1]
        Z = xt0[2]
        
        phi_t = S*0.5*self.bs + I*0.5*self.bi + Z*0.5*self.bz

        Pz = 1 - math.exp(-1*self.gamma)  ### Probability of recovery.
        xi = ( (1 - self.delta**(self.tau + 1) )/(1 - self.delta) ) - ( (1 - ( self.delta*(1 - Pz))**(self.tau + 1)  )/( 1 - self.delta*(1-Pz) ) ) ### Xi function in notes
        vti = ((0.25* (self.bz)**2 )**self.gamma1 - self.az)*xi ### Value of V_{t+1}(i) for t inside interval [t_0,t_0+tau-1)

        ### Probability that an s-type individual becomes infected at time t.
        ### Depends on selection of C_s^t

        def P_it(C_st):
            P_it = 1 - math.exp(-1*(0.5*self.beta*self.bi*I*C_st)/phi_t)
            return P_it

        """
        Now we employ the backwards induction method to compute C_t^s.
        - The idea is that the individual will become infected at time tau.
        - We start with time t+tau and move backwards to time t
        """

        def vs1(C_st, vti):
            expr0 = 0.5*self.beta*self.bi*I*math.exp(-1*(0.5*self.beta*self.bi*I*C_st)/phi_t)/phi_t
            expr1 = (self.gamma1*(self.bs*C_st - C_st*2)*(self.gamma1 - 1)*(self.bs - 2*C_st)) / expr0
            expr2 = (1 - P_it(C_st))*expr1 + P_it(C_st)*vti
            return (self.bs*C_st - C_st*2)*self.gamma1 - self.as1 - self.delta*expr2

        #print(f"for t={t} we compute C^s_opt")
        #start=time.time()
        C_st_array = np.linspace(0, 0.5*self.bs, 1000)
        C_st_args = [0]*(self.tau + 1)
        Vs1s = [0]*(self.tau + 1) ### length is tau+1 [goes from 0 to tau]
        Vi1s = [vti]*(self.tau - 1) + [(0.25* (self.bi)**2 )**self.gamma1 - self.ai] + [0]  ###length is tau+1

        ### t+tau
        C_st_tau_step = [vs1(C_st, vti = Vi1s[self.tau] ) for C_st in C_st_array]
        Vs1s[self.tau] = max(C_st_tau_step) ### This is V_{t_0+tau + 1}
        C_st_args[self.tau] = C_st_array[np.argmax(C_st_tau_step)]

        ### Go over all the other steps (backwards):
        for j in range(1, self.tau + 1):

            ### Get V_{t_0 + tau - j + 1}(i)
            v_i_tau_j_1=Vi1s[self.tau - j + 1]
            ### Get V_{t_0 + tau - j + 1}(s)
            v_s_tau_j_1=Vs1s[self.tau - j + 1]

            ### Use formula (6) of article to find V_{t_0+ tau -j}
            val_func_values = [ (self.bs*C_st - C_st**2)**self.gamma1 - self.as1 + 
            self.delta*((1 - P_it(C_st))*v_s_tau_j_1 + P_it(C_st)*v_i_tau_j_1) 
            for C_st in C_st_array]
            Vs1s[self.tau - j] = max(val_func_values)
            C_st_args[self.tau - j] = C_st_array[np.argmax(val_func_values)]

        Cs_opt = C_st_args[0]

        return Cs_opt

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

            self.logger.info(f"Solving {t}")

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
