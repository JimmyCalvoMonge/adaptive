import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import itertools
import time
from tqdm import tqdm
import sympy as sp
x = sp.var('x')
from sympy import Symbol
from sympy.solvers import solve
import math
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

# Import our adaptive module:
from non_adaptive import NonAdaptive

def get_coefficients_cubic(Rphi, Rmu, R0, kappa, theta):

    ### Returns the coefficients x3,x2,x1,x0 of the cubic polynomial in the notes.

    """
    kappa := C^i/C^s
    theta := C^z/C^s
    """

    x_3 = (Rphi**2)*(R0) + Rmu*(Rphi**2)*(kappa -1)
    
    x_2 = Rphi*(R0*(1 - Rphi) + Rmu*(R0 + Rphi))
    x_2 = x_2 + Rphi*(Rmu*(1 - Rmu)*(theta - 1) + Rmu*(1 + Rmu)*(kappa - 1)) 

    x_1 = Rmu*(R0*(1 - Rphi)  + Rphi*(1 - R0)  + Rmu*Rphi)
    x_1 = x_1+ Rmu*((1 - Rmu)*(theta - 1) + Rmu*(kappa - 1))

    x_0 = (Rmu**2)*(1 - R0)

    return [x_0,x_1,x_2,x_3]

def nmbr_rts_interval(Rphi, Rmu, val_tuple):

    kappa = val_tuple[0]
    theta = val_tuple[1]
    
    print(f"Processing tuple {(kappa,theta)}.")
    start_tuple = time.time()

    R0s = list(np.linspace(0.0001,2,1000))
    vals = []

    for R0 in tqdm(R0s):

        # Compute number of roots using Sturm's Theorem

        coefficients = get_coefficients_cubic(Rphi, Rmu, R0, kappa, theta)
        pol = sum([coefficients[i]*x**i for i in range(len(coefficients))])
        sturm_seq = sp.sturm(pol) # sturm sequence

        values_at_start = [float(sp.Poly(pol_sturm,x).eval(0)) for pol_sturm in sturm_seq]
        values_at_end = [float(sp.Poly(pol_sturm,x).eval(1)) for pol_sturm in sturm_seq]

        count_start = len(list(itertools.groupby(values_at_start, lambda values_at_start: values_at_start > 0)))
        count_end = len(list(itertools.groupby(values_at_end, lambda values_at_end: values_at_end > 0)))

        ans = count_start - count_end
        vals.append(ans)

    answer = max(vals)
    return answer

def evaluate_cubic(i, Rphi, Rmu, R0, kappa, theta):
    [x_0,x_1,x_2,x_3] = get_coefficients_cubic(Rphi, Rmu, R0, kappa, theta)
    return x_3*(i**3) + x_2*(i**2) + x_1*i + x_0

def solve_polynomial(Rphi, Rmu, R0, kappa, theta):
    
    x = Symbol('x')
    resp = solve(evaluate_cubic(x, Rphi, Rmu, R0, kappa, theta), x)
    resp = [(float(expr.as_real_imag()[0]), float(expr.as_real_imag()[1]) ) for expr in resp]
    resp = [expr[0] for expr in resp if expr[0]>0 and abs(expr[1])<1e-15]
    
    return resp

def get_convergence_point(
    N, recovered, prop_infected,
    mu, gamma, beta, phi,
    Cs, Ci, Cz,
    t_max):
    
    initial_infected = math.floor(prop_infected*N)
    x00 = [N - initial_infected - recovered, initial_infected, recovered]
    non_adaptive = NonAdaptive(mu, gamma, beta, phi, Cs, Ci, Cz, x00, t_max, dt=0.1)
    non_adaptive.solve_odes_system(method='odeint')
    equ_attached = non_adaptive.I[len(non_adaptive.I) - 1]/sum(x00)
    return equ_attached