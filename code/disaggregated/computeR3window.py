import numpy as np
import pandas as pd
import sympy as sp
x = sp.var('x')
from sympy import Symbol
from sympy.solvers import solve
from functools import partial
import multiprocessing
from tqdm import tqdm
import itertools

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


def evaluate_cubic(i, Rphi, Rmu, R0, kappa, theta):
    [x_0,x_1,x_2,x_3] = get_coefficients_cubic(Rphi, Rmu, R0, kappa, theta)
    return x_3*(i**3) + x_2*(i**2) + x_1*i + x_0


def solve_polynomial(Rphi, Rmu, R0, kappa, theta):
    
    x = Symbol('x')
    resp = solve(evaluate_cubic(x, Rphi, Rmu, R0, kappa, theta), x)
    resp = [(float(expr.as_real_imag()[0]), float(expr.as_real_imag()[1]) ) for expr in resp]
    resp = [expr[0] for expr in resp if expr[0]>0 and abs(expr[1])<1e-15]
    
    return resp


def r3window_comb(comb, Rphi, Rmu):

    r3windowdf = pd.DataFrame({})
    R0s = np.linspace(1, 1.06, 200)
    kappa = comb[0]
    theta = comb[1]
    
    for R0 in tqdm(R0s):
        try:
            
            # With solve:
            # roots = solve_polynomial(Rphi, Rmu, R0, kappa, theta)
            # num_roots = len(roots)

            # With Sturm:
            coefficients = get_coefficients_cubic(Rphi, Rmu, R0, kappa, theta)
            pol = sum([coefficients[i]*x**i for i in range(len(coefficients))])
            sturm_seq = sp.sturm(pol) # sturm sequence

            values_at_start = [float(sp.Poly(pol_sturm,x).eval(0)) for pol_sturm in sturm_seq]
            values_at_end = [float(sp.Poly(pol_sturm,x).eval(1)) for pol_sturm in sturm_seq]

            count_start = len(list(itertools.groupby(values_at_start, lambda values_at_start: values_at_start > 0)))
            count_end = len(list(itertools.groupby(values_at_end, lambda values_at_end: values_at_end > 0)))
            num_roots = count_start - count_end

            r3windowdf = r3windowdf.append(pd.DataFrame({
                'kappa': [kappa],
                'theta': [theta],
                'R0': [R0],
                'num_roots':[num_roots]
            }), ignore_index=True)

        except:
            pass
    
    print(f"Comb {comb[2]} done")

    return r3windowdf


if __name__ == '__main__':

    # Initial parameters
    mu = 0.00015
    gamma = 0.0027
    beta = 0.00096
    phi = 0.0044

    # R values:
    Rmu = mu/(mu+ gamma)
    Rphi = phi/(mu + gamma)

    print(f"""
    Parameters:
    mu : {mu}
    gamma: {gamma}
    beta : {beta}
    phi: {phi}
    Rmu = {Rmu}
    Rphi = {Rphi} 
    Multiple roots condition satisfied:
    {Rphi - (Rmu**2 + 1)/(Rmu**2 - Rmu + 1) > 0}
    """)

    kappas = np.linspace(0.05,1,20)
    thetas = np.linspace(0.05,2,40)

    # As DataFrame
    count = 0
    combinations = []
    for kappa in kappas:
        for theta in thetas:
            count = count + 1 
            combinations.append((round(kappa,2),round(theta,2), count))

    print(f"Total combinations: {len(combinations)}")

    print(" --- Start with multiprocessing ---")
    with multiprocessing.Pool(5) as pool:
        results = pool.map(partial(r3window_comb, Rphi=Rphi, Rmu=Rmu), combinations)

    # print('multiprocessing with tqdm?')
    # results = process_map(partial(r3window_comb, Rphi=Rphi, Rmu=Rmu),
    # combinations, max_workers=5, chunksize=4)

    print(f"Results done. Length: {len(results)}")
    final_df = pd.concat(results)
    final_df.to_csv("r3windowdf.csv", index=False)
    print("Done.")