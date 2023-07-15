"""

Find number of equilibria points found for each combination of
(C^i/C^s, C^z/C^s) at each R0.

Using Sturms theorem : https://en.wikipedia.org/wiki/Sturm%27s_theorem
For each ((C^i/C^s, C^z/C^s, R0) combination we form the equilibrium polynomial at the

Initial experiments for article
`A nonlinear relapse model with disaggregated contact rates: analysis of a forward-backward bifurcation`

"""

import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
import time
import sympy as sp
import itertools
import os
from base_path import base_path
x = sp.var('x')
max_contacts_susc = 20
adaptive_folder = f'{base_path}/disaggregated'

def get_coefficients_cubic(Rphi, Rmu, R0, kappa, xi):

    ### Returns the coefficients x3,x2,x1,x0 of the cubic polynomial in the notes.

    """
    kappa := C^i/C^s
    xi := C^z/C^s
    """

    x_3 = (Rphi**2)*(R0) + Rmu*(Rphi**2)*(kappa -1)
    
    x_2 = Rphi*(R0*(1 - Rphi) + Rmu*(R0 + Rphi))
    x_2 = x_2 + Rphi*( Rmu*(1 - Rmu)*( xi - 1) + Rmu*(1 + Rmu)*( kappa - 1) ) 

    x_1 = Rmu*(R0*(1 - Rphi)  + Rphi*(1 - R0)  + Rmu*Rphi )
    x_1 = x_1+ Rmu*( (1-Rmu)*( xi - 1 ) + Rmu*( kappa - 1 ) )

    x_0 = (Rmu**2)*(1-R0)

    return [x_0,x_1,x_2,x_3]

def nmbr_rts_interval(Rphi, Rmu, beta, mu, gamma, val_tuple):

    kappa = val_tuple[0]
    xi = val_tuple[1]

    if not f"res_{kappa}_{xi}.csv" in os.listdir(f"{adaptive_folder}/data/bifurcation_heatmap"):

        print(f"Processing tuple {(kappa,xi)}.")
        start_tuple = time.time()

        css = list(np.linspace(0.01,max_contacts_susc,1000)) #C^s
        cis = [cs*kappa for cs in css] #C^i = kappa C^s
        R0s = [ci*beta/(mu + gamma) for ci in cis] #R0
        vals = []

        for R0 in tqdm(R0s):

            # Compute number of roots using Sturm's Theorem
            coefficients = get_coefficients_cubic(Rphi, Rmu, R0, kappa, xi)
            pol = sum([coefficients[i]*x**i for i in range(len(coefficients))])
            sturm_seq = sp.sturm(pol) # sturm sequence
            
            values_at_start = [float(sp.Poly(pol_sturm,x).eval(0)) for pol_sturm in sturm_seq]
            values_at_end = [float(sp.Poly(pol_sturm,x).eval(1)) for pol_sturm in sturm_seq]
            
            count_start = len(list(itertools.groupby(values_at_start, lambda values_at_start: values_at_start > 0)))
            count_end = len(list(itertools.groupby(values_at_end, lambda values_at_end: values_at_end > 0)))
            
            ans = count_start - count_end
            vals.append(ans)

        answer = max(vals)
        
        tuple_data = pd.DataFrame({
            'kappa':[kappa],
            'xi':[xi],
            'numbr_roots':[answer]
        })
        tuple_data.to_csv(f"{adaptive_folder}/data/bifurcation_heatmap/res_{kappa}_{xi}.csv")

        end_tuple = time.time()
        print(f"Tuple {(kappa,xi)} DONE. This took {end_tuple - start_tuple} seconds. The answer: {answer}.")

if __name__ == '__main__':

    ### Initial parameters
    mu = 0.00015
    gamma = 0.0027
    beta = 0.009
    phi = 0.0044

    ### R values:
    Rmu= mu/(mu+ gamma)
    Rphi= phi/(mu + gamma)

    # values 
    kappas = [round(t,2) for t in np.linspace(0.1, 1, 30)]
    xis = [round(t,2) for t in np.linspace(0.1, 3, 30)]
    param_lists = [kappas, xis]
    param_grid = list(itertools.product(*param_lists))

    print(f"Number of tuples to process: {len(param_grid)}")
    print(f"Sequentially this will take approx {len(param_grid)/(60*60)} hours.")
    print("We start processing them: ==========================================")

    pool = mp.Pool()
    func = partial(nmbr_rts_interval, Rphi, Rmu, beta, mu, gamma)
    pool.map(func, param_grid)
    pool.close()
    pool.join()

    # Join all results:

    final_data = pd.DataFrame({})
    for file in os.listdir(f'{adaptive_folder}/data/bifurcation_heatmap'):
        if file.endswith(".csv") and file.startswith("res"):
            data_tuple = pd.read_csv(f'{adaptive_folder}/data/bifurcation_heatmap/{file}')
            final_data = final_data.append(data_tuple, ignore_index = True)
            os.remove(f'{adaptive_folder}/disaggregated/data/bifurcation_heatmap/{file}') # Delete file

    final_data.to_csv(f"{adaptive_folder}/disaggregated/data/bifurcation_heatmap/simulation_11_Nov_02.csv")
    print(f"We are done. Final data shape: {final_data.shape} ==========================")
