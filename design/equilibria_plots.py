### Adaptive module
from adaptive import Adaptive
import numpy as np
import pandas as pd
import plotly.express as px
from tqdm import tqdm
import multiprocessing as mp
from multiprocessing import Process, Manager
from sympy import Symbol
from sympy.solvers import solve
import base64
import kaleido
from functools import partial
import plotly.io as pio
pio.kaleido.scope.mathjax = None
adaptive_folder='C:/Users/jimmy/OneDrive/Desktop/Maestria Metodos Matematicos y Aplicaciones/Tesis/adaptive'

def get_i_opts(Rmu, Rphi, mu, gamma, beta, values):

    kappa=values[0]
    xi=values[1]

    print(f"Computing ({kappa},{xi})  ... ")

    css= list(np.linspace(0.0001,1,1000))
    cis= [cs*kappa for cs in css]
    czs= [cs*xi for cs in css]
    R0s= [ci*beta/(mu+gamma) for ci in cis]

    values_list=[]

    for j in tqdm(range(len(css))):

        cs=css[j]
        ci=cis[j]
        cz=czs[j]
        R0=R0s[j]

        roots = solve_polynomial(cs,ci,cz,R0,Rphi,Rmu)
        for root in roots:
            if abs(root)<=1:
                values_list.append({"R0":R0,"i_opt":root})

    df_values=pd.DataFrame({
            'R0': [item['R0'] for item in values_list],
            'i_opt':[item['i_opt'] for item in values_list]
    })

    R0_counts=df_values['R0'].value_counts().tolist()

    try:
        num_max_eq_pts=max(R0_counts)
    except:
        num_max_eq_pts=0

    df_final=pd.DataFrame({
        'C^i/C^s':[kappa],
        'C^z/C^s':[xi],
        'num_max_eq_pts':[num_max_eq_pts]
    })
    df_final.to_csv(f"{adaptive_folder}/data/bifurcation_heatmap/bifurcation_{kappa}_{xi}",index=False)
    return df_final

### Examples

def get_coefficients_cubic(cs,ci,cz,R0,Rphi,Rmu):
    
    ### Returns the coefficients x3,x2,x1,x0 of the cubic polynomial in the notes.

    x_3=(Rphi**2)*(R0) + Rmu*(Rphi**2)*( (ci/cs) -1)

    x_2=Rphi*(R0*(1-Rphi) + Rmu*(R0+Rphi))
    x_2= x_2 + Rphi*( Rmu*(1-Rmu)*( (cz/cs) - 1) + Rmu*(1+Rmu)*( (ci/cs) - 1) ) 

    x_1= Rmu*(R0*(1-Rphi)  + Rphi*(1-R0)  + Rmu*Rphi )
    x_1= x_1+ Rmu*( (1-Rmu)*( (cz/cs) -1 ) + Rmu*( (ci/cs) -1  )   )

    x_0=(Rmu**2)*(1-R0)

    return [x_0,x_1,x_2,x_3]

def evaluate_cubic(i,cs,ci,cz,R0,Rphi,Rmu):
    [x_0,x_1,x_2,x_3] = get_coefficients_cubic(cs,ci,cz,R0,Rphi,Rmu)
    return x_3*(i**3) + x_2*(i**2) + x_1*i + x_0

def solve_polynomial(cs,ci,cz,R0,Rphi,Rmu):
    
    x = Symbol('x')
    resp = solve(evaluate_cubic(x,cs,ci,cz,R0,Rphi,Rmu), x)
    resp = [(float(expr.as_real_imag()[0]), float(expr.as_real_imag()[1]) ) for expr in resp]
    resp = [expr[0] for expr in resp if expr[0]>0 and abs(expr[1])<1e-15]
    
    return resp

if __name__ == '__main__':

    ### Initial parameters
    mu = 0.00015
    gamma = 0.0027
    beta = 0.009
    phi = 0.0044

    ### R values:
    Rmu= mu/(mu+ gamma)
    Rphi= phi/(mu + gamma)

    ### Compute info for C^i/C^s = kappa and C^z/C^s=xi,
    ### Where kappa goes from 0.1 to 1.0 and xi goes from 0.1 to 2.0

    rang_kappa=[round(t,2) for t in np.linspace(0.1,1,50)]
    rang_xi=[round(t,2) for t in np.linspace(0.1,2,50)]

    iterable_list=[]
    for kappa in rang_kappa:
        for xi in rang_xi:
            iterable_list.append((kappa,xi))

    pool = mp.Pool()
    func = partial(get_i_opts, Rmu, Rphi, mu, gamma, beta)
    pool.map(func, iterable_list)
    pool.close()
    pool.join()