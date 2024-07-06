import sys
sys.path.append("../../")

import numpy as np
import matplotlib.pyplot as plt
from src.utils import compute_error
from src.metropolis_hastings import run_MH
from src.proj_langevin import run_PL, gen_init_point
from src.data_generation import generate_data
from src.utils import dump_run_information_from_tensors
from src.plotting import lighten_color

import time
"""
Evolution of the accuracy wrt the number of shots for different values of lambda for the prob-estimator
"""
def run_experiment(savefig=True):

    test_run = False
    n = 3
    d = 2**n
    n_meas = d * d
    n_iter = 5000
    n_burnin = 1000
    rho_type="rank2"
    
    shots_range = "exp"

    if shots_range == "very_large":
        shots = range(1000, 21000, 1000)
        ext = "_verylarge"
    elif shots_range == "large":
        shots = range(500, 10500, 500)
        ext = "_large"
    elif shots_range == "exp":
        shots = np.logspace(2, 7, 20, True, 10, dtype=np.int64)
        ext = "_exp"
    else:
        shots = range(100, 2100, 100)
        ext = ""

    n_samples = 3
    lambdas = range(500, 5500, 500) # 10 values
    accs_pl = np.zeros((len(lambdas), len(shots), n_samples))
    accs_prob = np.zeros((len(lambdas), len(shots), n_samples))
    saved_res_pl = np.zeros((len(shots), n_samples))

    start = time.perf_counter()

    for k, lambda_ in enumerate(lambdas):
        for i, n_shots in enumerate(shots):
            for j in range(n_samples):
                time_since_start = time.perf_counter() - start
                seed = j + n_samples * i + len(shots) * n_samples * k
                rho_true, As, y_hat = generate_data(n, n_meas, n_shots, rho_type=rho_type, seed = seed)

                init_point = gen_init_point(d,d)
                if not test_run:
                    As_flat = np.zeros((n_meas, 2**n * 2**n), dtype = np.complex128)
                    for l in range(n_meas):
                        As_flat[l,:] = As[:,:,l].flatten(order="C")
                    _, rho_last_prob, _ = run_MH(n, n_meas, n_shots, rho_true, As_flat, y_hat, n_iter, n_burnin, seed=None, init_point=init_point, gamma=lambda_)
                    err_prob = compute_error(rho_last_prob, rho_true)
                else:
                    err_prob = seed
                if k == 0:
                    if not test_run:
                        _, rho_avg_pl, _  = run_PL(n, n_meas, n_shots, rho_type, As, y_hat, n_iter, n_burnin, seed=None, init_point=init_point)
                        err_pl = compute_error(rho_avg_pl, rho_true)
                    else:
                        err_pl = seed + 1
                    saved_res_pl[i, j] = err_pl
                else:
                    err_pl = saved_res_pl[i, j]
                accs_prob[k,i,j] = err_prob
                accs_pl[k,i,j] = err_pl

    avg_accs_prob = accs_prob.mean(axis=2)
    avg_accs_pl = accs_pl.mean(axis=2)
    plt.figure()
    plt.loglog(shots, avg_accs_pl[0], label="langevin", c="orange")
    main_color_prob = "b"
    colors_prob = [lighten_color(main_color_prob, amount=s) for s in np.linspace(1.6, 0.2, len(lambdas))]
    for i in range(len(lambdas)):
        plt.loglog(shots, avg_accs_prob[i], label=rf"prob,$\lambda$={lambdas[i]}", c=colors_prob[i])
    plt.legend()
    plt.xlabel("Number of shots [#]")
    plt.ylabel("$L_2$ squared error")
    plt.title("Comparison of accuracy wrt shots, different lambda values, loglog")
    if savefig:
        plt.savefig(f"shots_acc_comp_shots{ext}_lambda_prob_loglog.pdf", bbox_inches="tight")
    plt.show()

    # 
    if savefig:
        dump_run_information_from_tensors(accs_prob, accs_pl, {"lambda": lambdas, "shots": shots, "sample": list(range(n_samples))}, "run_shots_lambda_prob")

if __name__ == "__main__":
    run_experiment()