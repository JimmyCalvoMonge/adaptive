import numpy as np
import pandas as pd
import itertools
import time
import multiprocessing as mp
from functools import partial
import os

adaptive_folder = 'C:/Users/jimmy/OneDrive/Desktop/Maestria Metodos Matematicos y Aplicaciones/Tesis/adaptive'
adaptive_folder = f'{adaptive_folder}/code/adaptive/data/bifurcation_heatmap_adaptive'
# adaptive_folder = './'

import warnings
warnings.filterwarnings("ignore")

# Import our adaptive module:
from adaptive_MDP import Adaptive

# Initial parameters Our example with relapse (from chapter 2)----- #
mu = 0.00015
gamma = 0.0027
beta = 0.00096
phi = 0.0044

# R values:
Rmu = mu/(mu+ gamma)
Rphi = phi/(mu + gamma)

max_contacts_susc = 20
N = 10000

kappa = 0.8
theta = 1.2

pre_index = ''
css = list(np.linspace(0.01, max_contacts_susc, 20)) #C^s
cis = [cs*kappa for cs in css] #C^i = kappa C^s
czs = [cs*theta for cs in css] #C^z = theta C^s
R0s = [ci*beta/(mu + gamma) for ci in cis] #R0
R0s = [round(r0, 4) for r0 in R0s]


def get_convergence_point(mu, gamma, beta, phi, kappa, theta, val_tuple):

    prop = val_tuple[0]
    R0_index = val_tuple[1]
    R0 = R0s[R0_index]

    b_s = 2*css[R0_index]
    b_i = 2*cis[R0_index]
    b_z = 2*czs[R0_index]

    """
    Simulate the model for mu, gamma, beta, phi, kappa and theta.
    Using different C_opt^{s*}, making C_opt^{i*} vary and also R_{0,opt}^*
    """

    if not f"res{pre_index}_{kappa}_{theta}_{prop}_{R0}.csv" in os.listdir(adaptive_folder):

        print(f"Processing tuple {(kappa, theta, prop, R0)} ...")
        start_tuple = time.time()

        x00 = [N - int(prop*N), int(prop*N), 0]
        final_conv = []
        final_stopping_point = []
        R0s_used = []

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
        steps = 50

        # Initialize adaptive instances
        instance_adaptive = Adaptive(
            mu, gamma, beta, phi,
            tau, delta,
            u_s, u_i, u_z,
            t_max, steps, x00, max_contacts=30,
            logs=False, verbose=False, tqdm=False,
            compute_max_t_threshold=1e-5)
        
        try:
            instance_adaptive.patch_uni_solutions()
            final_I = instance_adaptive.I[-1]/N
            final_stopping_point.append(instance_adaptive.stopping_point)
            final_conv.append(final_I)
            R0s_used.append(R0)
            message = 'ok'
        
        except Exception as e:
            message = e
            final_stopping_point.append(0)
            final_conv.append(0)
            R0s_used.append(f"{R0}_ERROR")
            pass

        data_this_iter = pd.DataFrame({
            'R0': R0s_used,
            'final_conv_inf_point': final_conv,
            'final_stopping_point': final_stopping_point
        })

        data_this_iter['kappa'] = kappa
        data_this_iter['theta'] = theta
        data_this_iter['prop'] = prop
        data_this_iter.to_csv(f"{adaptive_folder}/res{pre_index}_{kappa}_{theta}_{prop}_{R0}.csv")

        end_tuple = time.time()
        print(f"Tuple {(kappa, theta, prop, R0)} DONE. This took {(end_tuple - start_tuple)/60} minutes. Message: {message}")

    else:
        print(f"Tuple {(kappa, theta, prop, R0)} ALREADY PROCESSED. DONE.")


if __name__ == '__main__':

    print(f"Number of physical cores available: {mp.cpu_count()}")
    nmbr_proc = 0
    for file in os.listdir(adaptive_folder):
        if f'res{pre_index}_{kappa}_{theta}' in file:
            nmbr_proc = nmbr_proc + 1
    print(f"Number of processed tuples so far: {nmbr_proc} <--- ")

    # Bifurcation plot for (kappa, theta) using adaptive behavior.
    # In this case we don't know the final equilibrium points.

    prop_list = np.linspace(0.001, 0.25, 10)
    prop_list = [round(pg, 4) for pg in prop_list]
    param_lists = [prop_list, range(len(R0s))]
    param_grid = list(itertools.product(*param_lists))

    print(f"Total number of computations to perform: {len(param_grid)}")

    pool = mp.Pool(processes=mp.cpu_count() - 1)
    func = partial(get_convergence_point, mu, gamma, beta, phi, kappa, theta)
    pool.map(func, param_grid)
    pool.close()
    pool.join()

    # Join all results:
    final_data = pd.DataFrame({})
    for file in os.listdir(adaptive_folder):
        if file.endswith(".csv") and file.startswith(f"res{pre_index}_{kappa}_{theta}"):
            data_tuple = pd.read_csv(f'{adaptive_folder}/{file}')
            final_data = final_data.append(data_tuple, ignore_index = True)
            # os.remove(f'{adaptive_folder}/data/bifurcation_heatmap_adaptive/{file}') # Delete file

    final_data.to_csv(f"{adaptive_folder}/simulation_adaptive_bifurcation{pre_index}_{kappa}_{theta}.csv")
    print(f"We are done. Final data shape: {final_data.shape} ==========================")


