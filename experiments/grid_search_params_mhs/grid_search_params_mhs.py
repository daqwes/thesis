import sys
sys.path.append("../../")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.utils import compute_error, dump_run_information_from_tensor
from src.mh_studentt_prior import run_MH_studentt
from src.data_generation_sep import generate_data_exact_PL
from src.proj_langevin import gen_init_point
"""
Compare and evaluate the different proposals we could use for MH used with a student-t prior, with various scaling coefs.
"""
def run_experiment(savefig=True):
    seed = 0
    n = 3
    d= 2**n
    n_meas = 3**n
    n_shots = 2000
    n_iter = 5000
    n_burnin = 1000
    run_avg = True
    use_prop_in_ratio = False
    log_transform = True
    rho_true, As, y_hat = generate_data_exact_PL(n, n_meas, n_shots, rho_type="rank2", seed= seed)
    init_point = gen_init_point(d, d)

    proposals = [("normal_dep", 0.001), ("exp_dep", 0.1)]
    lambdas = [1e1, 1e2, 5e2, *(list(np.arange(1e3, 1.1e4, 1e3))), 2e4, 1e5]
    thetas = np.logspace(-6, 1, 1 - (-6) + 1, base=10, dtype="float")
    n_samples = 5
    accs_mhs = np.zeros((len(proposals), len(lambdas), len(thetas), n_samples))
    avgs = []
    for (i, (prop, prop_scale_coef)) in enumerate(proposals):
        for j, lambda_ in enumerate(lambdas):
            for k, theta in enumerate(thetas):
                avg_err = 0
                for sample in range(n_samples):
                    rhos_mh_stt, rho_mh_stt, cum_times_mh_stt, acc_rate  = run_MH_studentt(n, n_shots, As, y_hat, n_iter, n_burnin, seed = None, run_avg=run_avg, proposal_dist=prop, scaling_coef_prop=prop_scale_coef, use_prop_in_ratio=use_prop_in_ratio, log_transform=log_transform, init_point=init_point, lambda_ = lambda_, theta=theta)
                    err = compute_error(rho_mh_stt, rho_true)
                    accs_mhs[i,j,k,sample] = err
                    avg_err += err
                    print(prop, lambda_, theta, sample, err)
                avg_err/= n_samples
                avgs += [avg_err]*n_samples
                # print("Avg: ",prop, lambda_, theta, sample, avg_err)
                # print()
    if savefig:
        dump_run_information_from_tensor("mhs", accs_mhs, {"prop": [p[0] for p in proposals], "lambda": lambdas, "theta": thetas, "sample": list(range(n_samples))}, path="grid_search_params_mhs", avgs=avgs)

if __name__ == "__main__":
    run_experiment()
