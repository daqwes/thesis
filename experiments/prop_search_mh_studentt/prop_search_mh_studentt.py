import sys
sys.path.append("../../")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.utils import compute_error
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
    run_avg = False
    use_prop_in_ratio = False
    log_transform = True
    rho_true, As, y_hat = generate_data_exact_PL(n, n_meas, n_shots, rho_type="rank2", seed= seed)
    init_point = gen_init_point(d, d)

    proposals = ["normal_dep", "exp_dep", "normal"]
    scaling_coefs_prop = [0.001, 0.01, 0.05, 0.1, 0.5, 1, 10]
    reps = 3
    results = []
    for prop in proposals:
        for coef in scaling_coefs_prop:
            avg_err = 0
            avg_acc_rate = 0
            for i in range(reps):
                rhos_mh_stt, rho_mh_stt, cum_times_mh_stt, acc_rate  = run_MH_studentt(n, n_shots, As, y_hat, n_iter, n_burnin, seed = None, run_avg=run_avg, proposal_dist=prop, scaling_coef_prop = coef, use_prop_in_ratio=use_prop_in_ratio, log_transform=log_transform, init_point=init_point)
                err = compute_error(rho_mh_stt, rho_true)
                avg_err += err
                avg_acc_rate += acc_rate
            avg_err /= reps
            avg_acc_rate /= reps
            res = {"prop": prop, "coef": coef, "err": avg_err, "acc_rate": avg_acc_rate}
            results.append(res)

    results_df = pd.DataFrame.from_records(results)
    results_df.sort_values(by="err")
    results_df.to_csv("prop_search_mh_studentt.csv", sep=',')

if __name__ == "__main__":
    run_experiment()
