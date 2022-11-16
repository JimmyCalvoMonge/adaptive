from scipy.integrate import odeint
import numpy as np
import time
import matplotlib.pyplot as plt
import math
from sympy.solvers import solve
from sympy import Symbol
import pandas as pd
import plotly.express as px

class Adaptive():
    
    def __init__(self,mu,gamma,beta,phi,bs,bi,bz,as1,ai,az,gamma1,tau,delta,t_max,x0):

        ### Initial parameters
        self.mu = mu
        self.gamma = gamma
        self.beta = beta
        self.phi = phi
        
        ### R values:
        self.Rmu= self.mu/(self.mu+ self.gamma)
        self.Rphi= self.phi/(self.mu + self.gamma)

        ### Adaptive parameters
        self.bi= bi
        self.bz= bz
        self.bs= bs
        self.ai= ai
        self.az= az
        self.as1= as1
        self.gamma1= gamma1
        self.tau= tau
        self.delta= delta

        ### Simulation parameters
        self.t_max= t_max
        self.h=t_max/1000
        self.t = np.linspace(0,self.t_max,int(self.t_max/self.h))
        self.x0= x0

        ### Adaptive System Values

        self.adaptive_values={
        't':[],
        'Cs':[],
        'Ci':[],
        'Cz':[],
        'S':[],
        'I':[],
        'Z':[],
        }

        self.non_adaptive_values={
        't':[],
        'Cs':[],
        'Ci':[],
        'Cz':[],
        'S':[],
        'I':[],
        'Z':[],
        }
        
    def initialize_csiz(self,cs,ci,cz):
        ###cs,ci,cz are functions

        self.cs=cs
        self.ci=ci
        self.cz=cz

    ### Solve non adaptive system (cs,ci,cz given as explicit functions by user)
    
    def state_odes_system(self, x, t, beta, mu, phi, gamma, cs, ci, cz):

        # assign each function to a vector element
        s = x[0]
        i = x[1]
        z = x[2]

        # C function 
        C = cs(t,s,i,z)*ci(t,s,i,z)/(s*cs(t,s,i,z) + i*ci(t,s,i,z) + z*cz(t,s,i,z) )

        ### Add values to store
        self.non_adaptive_values['t'] = self.non_adaptive_values['t']+[t]
        self.non_adaptive_values['Cs'] = self.non_adaptive_values['Cs']+[cs]
        self.non_adaptive_values['Ci'] = self.non_adaptive_values['Ci']+[ci]
        self.non_adaptive_values['Cz'] = self.non_adaptive_values['Cz']+[cz]
        self.non_adaptive_values['S'] = self.non_adaptive_values['S']+[s]
        self.non_adaptive_values['I'] = self.non_adaptive_values['I']+[i]
        self.non_adaptive_values['Z'] = self.non_adaptive_values['Z']+[z]
        
        # System 
        dSdt = -C*beta*s*i + mu - mu*s
        dIdt = C*beta*s*i + phi*z*i - (gamma+mu)*i  
        dZdt = gamma*i - phi*z*i - mu*z

        return [dSdt, dIdt, dZdt]

    def solve_odes_system_non_adaptive(self):
        
        x0= self.x0
        t= self.t
        beta= self.beta
        mu= self.mu
        phi= self.phi
        gamma= self.gamma
        cs= self.cs
        ci= self.ci
        cz= self.cz

        """
        Solve the classical system with initial conditions
        """

        x = odeint(self.state_odes_system, x0, t, args=(beta, mu, phi, gamma, cs, ci, cz))

        s = x[:,0]
        i = x[:,1]
        z = x[:,2]

        return s,i,z

    ### Find equilibrium with cubic from theory ###
    ### Using cs,ci,cz as constants ###

    def get_coefficients_cubic(self,cs,ci,cz,R0):

        ### These two are constant
        Rmu=self.Rmu
        Rphi=self.Rphi
        
        ### Returns the coefficients x3,x2,x1,x0 of the cubic polynomial in the notes.

        x_3=(Rphi**2)*(R0) + Rmu*(Rphi**2)*( (ci/cs) -1)

        x_2=Rphi*(R0*(1-Rphi) + Rmu*(R0+Rphi))
        x_2= x_2 + Rphi*( Rmu*(1-Rmu)*( (cz/cs) - 1) + Rmu*(1+Rmu)*( (ci/cs) - 1) ) 

        x_1= Rmu*(R0*(1-Rphi)  + Rphi*(1-R0)  + Rmu*Rphi )
        x_1= x_1+ Rmu*( (1-Rmu)*( (cz/cs) -1 ) + Rmu*( (ci/cs) -1  )   )

        x_0=(Rmu**2)*(1-R0)

        return [x_0,x_1,x_2,x_3]

    def evaluate_cubic(self,i,cs,ci,cz,R0):
        [x_0,x_1,x_2,x_3] = self.get_coefficients_cubic(cs,ci,cz,R0)
        return x_3*(i**3) + x_2*(i**2) + x_1*i + x_0

    def solve_polynomial(self,cs,ci,cz,R0):
        
        x = Symbol('x')
        resp = solve(self.evaluate_cubic(x,cs,ci,cz,R0), x)
        resp = [(float(expr.as_real_imag()[0]), float(expr.as_real_imag()[1]) ) for expr in resp]
        resp = [expr[0] for expr in resp if expr[0]>0 and abs(expr[1])<1e-15]
        
        return resp

    ### Bifurcation plot with cs,ci,cz given as explicit varying constants by user.

    def plot_equilibria_non_adaptive(self,css,cis,czs,R0s,title):

        """
        Plot equilibria found with cs's, ci's, cz's and therefore R0's varying.
        Bifurcation plot.
        R0 depends on the ci's
        """

        df_values_plot=pd.DataFrame({
            'R0':[],
            'i_opt':[]
        })

        start=time.time()
        
        for j in range(len(css)):

            cs=css[j]
            ci=cis[j]
            cz=czs[j]
            R0=R0s[j]
            
            roots=self.solve_polynomial(cs,ci,cz,R0)
            
            for root in roots:
                if abs(root)<=1:
                    df_values_plot=df_values_plot.append(pd.DataFrame({
                        'R0':[R0],
                        'i_opt':[root]
                    }),ignore_index=True)

        end=time.time()
        print(f"Computing Equilibria for kappas took {(end-start)/60} minutes.")

        fig = px.scatter(df_values_plot, x="R0", y="i_opt")
        fig.update_traces(marker=dict(size=7, line=dict(width=0.1, color='Black')))
        fig.update_layout(paper_bgcolor='#DDE1E2',plot_bgcolor='#FFFFFF',height=500, width=600, title=title)

        return fig

    ### Adaptive Functions ###

    def get_Cs_opt(self,S,I,Z):

        """
        Select the optimal C_t^s in the interval [0, 0.5b^s] for which we attain the maximum of the 
        value function. For now I'm using the direct implementation from equation (2.2) in the notes.
        We might need to review this and instead use the backwards induction technique.

        Attributes:
        - values of S_t,I_t,Z_t at time t.
        - adaptive utility parameters b^z,b^i,b^s,gamma.
        - time window planning info: tau, delta.
        - SIR model parameters: beta, gamma.
        - time: t
        """

        bi=self.bi
        bz=self.bz
        bs=self.bs
        ai=self.ai
        az=self.az
        as1=self.as1
        gamma1=self.gamma1
        tau=self.tau
        delta=self.delta
        beta=self.beta
        gamma=self.gamma

        phi_t = S*0.5*bs + I*0.5*bi + Z*0.5*bz

        Pz = 1 - math.exp(-1*gamma)  ### Probability of recovery.
        xi = ( (1 - delta**(tau+1) )/(1 - delta) ) - ( (1 - ( delta*(1-Pz) )**(tau+1)  )/( 1 - delta*(1-Pz) ) ) ### Xi function in notes
        vti = ((0.25* (bz)**2 )**gamma1 - az)*xi ### Value of V_{t+1}(i) for t inside interval [t_0,t_0+tau-1)

        ### Probability that an s-type individual becomes infected at time t.
        ### Depends on selection of C_s^t

        def P_it(C_st):
            P_it = 1 - math.exp(-1*(0.5*beta*bi*I*C_st)/phi_t)
            return P_it

        """
        Now we employ the backwards induction method to compute C_t^s.
        - The idea is that the individual will become infected at time tau.
        - We start with time t+tau and move backwards to time t
        """

        def vs1(C_st, vti):
            expr0 = 0.5*beta*bi*I*math.exp(-1*(0.5*beta*bi*I*C_st)/phi_t)
            expr1 = (gamma1*(bs*C_st - C_st**2)**(gamma1-1) + bs -2*C_st) / expr0
            expr2 = (1 - P_it(C_st))*expr1 + vti
            return (bs*C_st - C_st**2)**gamma1 - as1 - delta*expr2

        #print(f"for t={t} we compute C^s_opt")
        #start=time.time()
        C_st_array = np.linspace(0,0.5*bs,1000)
        C_st_args = [0]*(tau+1)
        Vs1s=[0]*(tau+1) ### length is tau+1 [goes from 0 to tau]
        Vi1s=[vti]*(tau - 1) + [(0.25* (bi)**2 )**gamma1 - ai] + [0]  ###length is tau+1

        ### t+tau
        C_st_tau_step = [vs1(C_st, vti = Vi1s[tau] ) for C_st in C_st_array]
        Vs1s[tau] = max(C_st_tau_step) ### This is V_{t_0+tau + 1}
        C_st_args[tau] = C_st_array[np.argmax(C_st_tau_step)]

        ### Go over all the other steps (backwards):
        for j in range(1,tau+1):

            ### Get V_{t_0 + tau - j + 1}(i)
            v_i_tau_j_1=Vi1s[tau-j+1]
            ### Get V_{t_0 + tau - j + 1}(s)
            v_s_tau_j_1=Vs1s[tau-j+1]

            ### Use formula (6) of article to find V_{t_0+ tau -j}
            val_func_values = [ (bs*C_st - C_st**2)**gamma1 - as1 + delta*((1 - P_it(C_st))*v_s_tau_j_1 + P_it(C_st)*v_i_tau_j_1) for C_st in C_st_array]
            Vs1s[tau-j] = max(val_func_values)
            C_st_args[tau-j] = C_st_array[np.argmax(val_func_values)]

        Cs_opt=C_st_args[0]
        #end=time.time()
        #print(f"for t={t} computing C^s_opt took {end-start} seconds")

        return Cs_opt

    def state_odes_system_adaptive(self,x,t):

        beta=self.beta
        mu=self.mu
        phi=self.phi
        gamma=self.gamma
        bi=self.bi 
        bz=self.bz

        # assign each function to a vector element
        s = x[0]
        i = x[1]
        z = x[2]

        cs=self.get_Cs_opt(s,i,z)
        ci=0.5*bi
        cz=0.5*bz

        # C function 
        C = cs*ci/(s*cs + i*ci + z*cz)

        # System 
        dSdt = -C*beta*s*i + mu - mu*s
        dIdt = C*beta*s*i + phi*z*i - (gamma+mu)*i  
        dZdt = gamma*i - phi*z*i - mu*z

        ### Add values to store
        self.adaptive_values['t'] = self.adaptive_values['t']+[t]
        self.adaptive_values['Cs'] = self.adaptive_values['Cs']+[cs]
        self.adaptive_values['Ci'] = self.adaptive_values['Ci']+[ci]
        self.adaptive_values['Cz'] = self.adaptive_values['Cz']+[cz]
        self.adaptive_values['S'] = self.adaptive_values['S']+[s]
        self.adaptive_values['I'] = self.adaptive_values['I']+[i]
        self.adaptive_values['Z'] = self.adaptive_values['Z']+[z]

        return [dSdt, dIdt, dZdt]

    def solve_odes_system_adaptive(self):
        
        t=self.t
        x0=self.x0

        """
        Solve the classical system with initial conditions
        """

        start=time.time()
        x = odeint(self.state_odes_system_adaptive, x0, t)
        end=time.time()
        print(f"Solving adaptive algorithm took {(end-start)/60} minutes.")
        s = x[:,0]
        i = x[:,1]
        z = x[:,2]

        return s,i,z

    def get_cubic_values_adaptive(self):

        """
        Once S,I,Z and Cs,Ci,Cz are obtained using the adaptive method,
        we evaluate f(i) to find roots.
        Note that the coefficients of f(i) depend on cs,ci,cz and these in turn
        depend on s,i,z.


        Remember that R0= beta/(gamma+mu)* [ lim_{(s,i,z)->(1,0,0)} ci(t,s,i,z) ]
        """

        # TODO FIX THIS To get value of C^i when it's not constant #
        R0= self.beta*self.adaptive_values['Ci'][0]/(self.mu + self.gamma)

        cubic_vals=[self.evaluate_cubic(
            self.adaptive_values['I'][i],
            self.adaptive_values['Cs'][i],
            self.adaptive_values['Ci'][i],
            self.adaptive_values['Cz'][i],
            R0) 
            for i in range(len(self.adaptive_values['t']))]

        return cubic_vals

    def plot_equilibria_adaptive(self):
        # TODO
        return None

    ### Plot a general solution ###

    def plot_ode_solution(self,S,I,Z,t,model):
        plt.plot(t,S,label="Susceptible")
        plt.plot(t,I,label="Infected")
        plt.plot(t,Z,label="Recovered")
        plt.title(f"Plot of S-I-Z functions ({model})")
        plt.xlabel("Time (t)")
        plt.ylabel("Number of individuals")
        plt.legend(loc="upper right")
        plt.rcParams["figure.figsize"] = (10,6)
        plt.show()

    def __str__():
        # TODO
        return "ODE System."
