### Adaptive module
from adaptive import Adaptive
import numpy as np
import pandas as pd
import plotly.express as px
import time
from multiprocessing import Process, Manager
from sympy import Symbol
from sympy.solvers import solve
import base64
import kaleido
import plotly.io as pio
pio.kaleido.scope.mathjax = None

### Examples\

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

def get_i_opt(L, chunk, Rmu, Rphi, css, cis, czs, R0s):

    for j in chunk:
        
        cs=css[j]
        ci=cis[j]
        cz=czs[j]
        R0=R0s[j]

        roots = solve_polynomial(cs,ci,cz,R0,Rphi,Rmu)
        
        for root in roots:
            if abs(root)<=1:
                L.append({"R0":R0,"i_opt":root})

def save_equilibria_plot(values_list,title,figname):

    df_values_plot=pd.DataFrame({
            'R0': [item['R0'] for item in values_list],
            'i_opt':[item['i_opt'] for item in values_list]
    })
    fig = px.scatter(df_values_plot, x="R0", y="i_opt")
    fig.update_traces(marker=dict(size=7, line=dict(width=0.1, color='Black')))
    fig.update_layout(paper_bgcolor='#DDE1E2',plot_bgcolor='#FFFFFF',height=500, width=600, title=title)
    fig.write_image(f'figure_{figname}.png',engine='kaleido')

if __name__ == '__main__':

    ### Initial parameters
    mu = 0.00015
    gamma = 0.0027
    beta = 0.009
    phi = 0.0044

    ### R values:
    Rmu= mu/(mu+ gamma)
    Rphi= phi/(mu + gamma)
    
    with Manager() as manager:

        rang=list(np.linspace(0.1,0.9,18))

        for nu in rang:
            for theta in rang:

                ### Define vector here:
                css= list(np.linspace(0.0001,0.6,700))
                cis= [cs*theta for cs in css]
                czs= [cs*(1-nu) for cs in css]
                R0s= [ci*beta/(mu+gamma) for ci in cis]

                chunk_nums=7
                chunk_sizes=int(700/chunk_nums)
                chunks=[list(range(i,i+chunk_sizes)) for i in range(0,700,chunk_sizes)]

                values_list = manager.list()

                print(f"Start computing points of bifurcation plot for Cs={theta}Ci and Cz=Cs(1-{nu}):")
                start=time.time()
                processes = [Process(
                    target=get_i_opt,
                    args=(values_list, chunk, Rmu, Rphi, css, cis, czs, R0s)) 
                    for chunk in chunks]

                for process in processes:
                    process.start()
                for process in processes:
                    process.join()
                end=time.time()
                print(f"This took {(end-start)/60} minutes")

                save_equilibria_plot(
                    values_list,
                    title=f'Ci = Cs*{theta} and Cz=Cs(1 - {nu})',
                    figname=f'Ci_equal_{theta}Cs_Cz_equal_Cs(1_minus_{nu})')