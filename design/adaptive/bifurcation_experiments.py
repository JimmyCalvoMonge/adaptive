import numpy as np
import pandas as pd
import itertools
import time
import multiprocessing as mp
from functools import partial
import os

adaptive_folder='C:/Users/jimmy/OneDrive/Desktop/Maestria Metodos Matematicos y Aplicaciones/Tesis/adaptive'

import warnings
warnings.filterwarnings("ignore")

# Import our adaptive module:
from adaptive_MDP import Adaptive_New

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

def get_convergence_point(mu, gamma, beta, phi, val_tuple, prop):

    kappa = val_tuple[0]
    theta = val_tuple[1]

    if not f"res_{kappa}_{theta}.csv" in os.listdir(f"{adaptive_folder}/data/bifurcation_heatmap_adaptive"):

        print(f"Processing tuple {(kappa, theta, prop)}.")
        start_tuple = time.time()

        css = list(np.linspace(0.01,max_contacts_susc,1000)) #C^s
        cis = [cs*kappa for cs in css] #C^i = kappa C^s
        czs = [cs*kappa for cs in css] #C^z = theta C^s
        R0s = [ci*beta/(mu + gamma) for ci in cis] #R0

        x00 = [N - int(prop*N), int(prop*N), 0]

        final_conv = []
        final_stopping_point = []
        R0s_used = []

        for i in range(len(R0s)):

            ### Adaptive parameters
            # Quadratic Utility functions:

            b_s = css[i] # Making the max utility attained at b_s/2 
            b_i = cis[i] # Making the max utility attained at b_i/2 
            b_z = czs[i] # Making the max utility attained at b_z/2 
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
            t_max = 7000
            steps = 100

            # Initialize adaptive instances
            instance_adaptive = Adaptive_New(
                mu, gamma, beta, phi,
                tau, delta,
                u_s, u_i, u_z,
                t_max, steps, x00, max_contacts=30,
                logs=False, verbose=False, tqdm=False,
                compute_max_t_threshold=1e-6)

            try:
                instance_adaptive.patch_uni_solutions()
                final_I = instance_adaptive.I[-1]/N
                final_stopping_point.append(instance_adaptive.stopping_point)
                final_conv.append(final_I)
                R0s_used.append(R0s[i])
            except Exception:
                pass

        data_this_iter = pd.DataFrame({
            'R0': R0s_used,
            'final_conv_inf_point': final_conv,
            'final_stopping_point': final_stopping_point
        })

        data_this_iter['kappa'] = kappa
        data_this_iter['theta'] = theta
        data_this_iter['prop'] = prop

        data_this_iter.to_csv(f"{adaptive_folder}/data/bifurcation_heatmap_adaptive/res_{kappa}_{theta}_{prop}.csv")

        end_tuple = time.time()
        print(f"Tuple {(kappa, theta, prop)} DONE. This took {end_tuple - start_tuple} seconds.")


if __name__ == '__main__':

    # values 
    # kappas = [round(t,2) for t in np.linspace(0.1, 1, 30)]
    # thetas = [round(t,2) for t in np.linspace(0.1, 3, 30)]
    # param_lists = [kappas, thetas]
    # param_grid = list(itertools.product(*param_lists))

    # Bifurcation plot for (kappa, theta) = (0.8, 1.7) using adaptive behavior.
    param_grid = np.linspace(0.00001, 0.05, 50)

    pool = mp.Pool()
    func = partial(get_convergence_point, mu, gamma, beta, phi, (0.8, 1.7))
    pool.map(func, param_grid)
    pool.close()
    pool.join()

    # Join all results:
    final_data = pd.DataFrame({})
    for file in os.listdir(f'{adaptive_folder}/data/bifurcation_heatmap'):
        if file.endswith(".csv") and file.startswith("res"):
            data_tuple = pd.read_csv(f'{adaptive_folder}/data/bifurcation_heatmap_adaptive/{file}')
            final_data = final_data.append(data_tuple, ignore_index = True)
            os.remove(f'{adaptive_folder}/data/bifurcation_heatmap_adaptive/{file}') # Delete file

    final_data.to_csv(f"{adaptive_folder}/data/bifurcation_heatmap_adaptive/simulation_28_Mar_2023.csv")
    print(f"We are done. Final data shape: {final_data.shape} ==========================")

