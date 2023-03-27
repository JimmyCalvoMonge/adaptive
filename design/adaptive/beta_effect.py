import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

# Import our adaptive module:
from adaptive_MDP import Adaptive_New
import multiprocessing as mp
from functools import partial

# ================================================== #

# Initial parameters (Example from Fenichel et al):
mu = 0
gamma = 0.1823
phi = 0

# R values:
Rmu = mu/(mu+ gamma)
Rphi = phi/(mu + gamma)

### Adaptive parameters
# Quadratic Utility functions:

b_s = 10 # Making the max utility attained at b_s/2 
b_i = 6.67 # Making the max utility attained at b_i/2 
b_z = 10 # Making the max utility attained at b_z/2 
a_s = 0
a_i = 1.826
a_z = 0
nu = 0.25
tau = 12
delta = 0.99986
max_contacts = 15

# ================================================== #

# Immediate rewards: (Utility functions)
def u_s(a):
    return (b_s*a - a**2)**nu - a_s
def u_i(a):
    return (b_i*a - a**2)**nu - a_i
def u_z(a):
    return (b_z*a - a**2)**nu - a_z

### Simulation parameters
t_max = 150
steps = 100
x00 = [9999, 1, 0]
betas = np.linspace(0.05, 0.1, 100)

def get_min_peaks(mu, gamma, tau, phi, delta, u_s, u_i, u_z,
                  t_max, steps, x00, max_contacts, beta):
    print(beta)
    instance_adaptive = Adaptive_New(
        mu, gamma, beta, phi,
        tau, delta,
        u_s, u_i, u_z,
        t_max, steps, x00, max_contacts,
        logs=False, verbose=False, tqdm=False)
    instance_adaptive.patch_uni_solutions()
    min_cts_s = np.nanmin(instance_adaptive.cs_history)
    peak_I = np.nanmax(instance_adaptive.I)

    return min_cts_s, peak_I, beta
    
if __name__ == '__main__':

    pool = mp.Pool(4)
    func = partial(get_min_peaks, mu, gamma, tau, phi, delta, u_s, u_i, u_z,
                    t_max, steps, x00, max_contacts)
    results = pool.map(func, betas)
    pool.close()
    pool.join()

    betas_list = []
    min_cts_list = []
    peak_prevs = []

    for i in range(len(results)):
        min_cts_list.append(results[i][0])
        peak_prevs.append(results[i][1])
        betas_list.append(results[i][2])

    data = pd.DataFrame({
        'beta' : betas_list,
        'min_cs': min_cts_list,
        'peak_i': peak_prevs
    })

    data.to_csv("./data/effect_of_beta_1.csv")

