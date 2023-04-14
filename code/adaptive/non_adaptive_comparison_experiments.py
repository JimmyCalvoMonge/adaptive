import numpy as np
import pandas as pd
import itertools
import time
import multiprocessing as mp
from functools import partial
import os

"""
Conjecture 1 Simulations
Vary: beta, phi, R_0 and initial conditions.
"""

adaptive_folder = 'C:/Users/jimmy/OneDrive/Desktop/Maestria Metodos Matematicos y Aplicaciones/Tesis/adaptive'
adaptive_folder = f'{adaptive_folder}/code/adaptive/data/conj_1_simulations'

if not os.path.exists(adaptive_folder):
    os.makedirs(adaptive_folder, exist_ok=True)

import warnings
warnings.filterwarnings("ignore")

# Import our adaptive module:
from adaptive_MDP import Adaptive

# Import non adaptive modules to compare:
import sys
sys.path.insert(0, 'C:/Users/jimmy/OneDrive/Desktop/Maestria Metodos Matematicos y Aplicaciones/Tesis/adaptive/code/disaggregated')
non_adaptive = __import__('non_adaptive', globals=None, locals=None, fromlist=(), level=0)
poly_utils = __import__('poly_utils', globals=None, locals=None, fromlist=(), level=0)
NonAdaptive = non_adaptive.NonAdaptive

# Initial parameters Our example with relapse (from chapter 2)----- #
mu = 0.00015
gamma = 0.0027

max_contacts_susc = 30
N = 10000

kappa = 0.5
theta = 1.2

css = list(np.linspace(0.01, max_contacts_susc, 5)) #C^s
cis = [cs*kappa for cs in css] #C^i = kappa C^s
czs = [cs*theta for cs in css] #C^z = theta C^s

def get_convergence_point_comparison(kappa, theta,
                          val_tuple):
    
    # tuple: (beta, phi, prop, R0)

    beta = val_tuple[0]
    phi = val_tuple[1]
    prop = val_tuple[2]
    R0_index = val_tuple[3]

    b_s = css[R0_index]
    b_i = cis[R0_index]
    b_z = czs[R0_index]
    R0 = round((b_i/2)*beta/(mu + gamma),4)

    """
    Simulate the model for mu, gamma, beta, phi, kappa and theta.
    Using different C_opt^{s*}, making C_opt^{i*} vary and also R_{0,opt}^*
    """

    if not f"res_{kappa}_{theta}_{beta}_{phi}_{prop}_{R0}.csv" in os.listdir(adaptive_folder):

        print(f"Processing tuple {(kappa, theta, prop, R0)} ...")
        start_tuple = time.time()

        x00 = [N - int(prop*N), int(prop*N), 0]

        ### Adaptive parameters
        # Quadratic Utility functions:

        a_s = 0
        a_i = 0
        a_z = 0
        nu = 0.01375
        tau = 12
        delta = 0.9986

        # Immediate rewards:
        def u_s(a):
            return (b_s*a - a**2)**nu - a_s
        def u_i(a):
            return (b_i*a - a**2)**nu - a_i
        def u_z(a):
            return (b_z*a - a**2)**nu - a_z

        ### Simulation parameters
        t_max = 10000 # max days
        steps = 100

        try:

            # Initialize adaptive instances
            instance_adaptive = Adaptive(
                mu, gamma, beta, phi,
                tau, delta,
                u_s, u_i, u_z,
                t_max, steps, x00, max_contacts=30,
                logs=False, verbose=False, tqdm=False,
                compute_max_t_threshold=1e-6)

            # Solve
            instance_adaptive.patch_uni_solutions()
            final_I_adaptive = instance_adaptive.I[-1]/N
            final_diff_adaptive = instance_adaptive.I[-1] - instance_adaptive.I[-steps]

            print("adaptive done")

            # Run Non Adaptive With Last Values from Adaptive Computation
            instance_non_adaptive_ex_post = NonAdaptive(
                mu, gamma, beta, phi,
                instance_adaptive.cs_history[-1],
                instance_adaptive.ci_history[-1],
                instance_adaptive.cz_history[-1], x00, t_max, tqdm=False)
            
            # Solve
            instance_non_adaptive_ex_post.solve_odes_system()
            final_I_non_adaptive = instance_non_adaptive_ex_post.I[-1]/sum(x00)
            final_diff_non_adaptive = instance_non_adaptive_ex_post.I[-1] - instance_non_adaptive_ex_post.I[-steps]

            print("non adaptive done")

            message = 'ok'

        except Exception as e:
            message = e

        if message != 'ok':
            final_I_adaptive = 0
            final_diff_adaptive = 0
            final_I_non_adaptive = 0
            final_diff_non_adaptive = 0
            pass

        data_this_iter = pd.DataFrame({
            'R0': [R0],
            'kappa': [kappa],
            'theta':[theta],
            'beta':[beta],
            'phi':[phi],
            'prop':[prop],
            'final_conv_inf_point_adaptive': [final_I_adaptive],
            'final_diff_adaptive': [final_diff_adaptive],
            'final_conv_inf_point_adaptive': [final_I_non_adaptive],
            'final_diff_non_adaptive' : [final_diff_non_adaptive],
            'message' : [message]
        })

        data_this_iter.to_csv(f"{adaptive_folder}/res_{kappa}_{theta}_{prop}_{R0}.csv")

        end_tuple = time.time()
        print(f"Tuple {(kappa, theta, prop, R0)} DONE. This took {(end_tuple - start_tuple)/60} minutes. Message: {message}")

    else:
        print(f"Tuple {(kappa, theta, prop, R0)} ALREADY PROCESSED. DONE.")


if __name__ == '__main__':

    print(f"Number of physical cores available: {mp.cpu_count()}")
    nmbr_proc = 0
    for file in os.listdir(adaptive_folder):
        if f'res_{kappa}_{theta}' in file:
            nmbr_proc = nmbr_proc + 1
    print(f"Number of processed tuples so far: {nmbr_proc} <--- ")

    # Bifurcation plot for (kappa, theta) using adaptive behavior.
    # In this case we don't know the final equilibrium points.

    beta_list = np.linspace(0.0001, 0.01, 5)
    phi_list = np.linspace(0.001, 0.1, 5)
    prop_list = np.linspace(0.001, 0.25, 5)

    beta_list = [round(b, 4) for b in beta_list]
    phi_list = [round(p, 4) for p in phi_list]
    prop_list = [round(pg, 4) for pg in prop_list]

    param_lists = [beta_list, phi_list, prop_list, range(len(css))]
    param_grid = list(itertools.product(*param_lists))

    param_grid = [param_grid[4]]
    print(f"Total number of computations to perform: {len(param_grid)}")

    pool = mp.Pool(processes=mp.cpu_count() - 1)
    func = partial(get_convergence_point_comparison, kappa, theta)
    pool.map(func, param_grid)
    pool.close()
    pool.join()

    # Join all results:
    final_data = pd.DataFrame({})
    for file in os.listdir(adaptive_folder):
        if file.endswith(".csv") and file.startswith(f"res_{kappa}_{theta}"):
            data_tuple = pd.read_csv(f'{adaptive_folder}/{file}')
            final_data = final_data.append(data_tuple, ignore_index = True)
            # os.remove(f'{adaptive_folder}/data/bifurcation_heatmap_adaptive/{file}') # Delete file

    final_data.to_csv(f"{adaptive_folder}/simulation_adaptive_non_adaptive_comparison_{kappa}_{theta}.csv")
    print(f"We are done. Final data shape: {final_data.shape} ==========================")


