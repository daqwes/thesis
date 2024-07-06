import sys
sys.path.append("../../")

import numpy as np
import matplotlib.pyplot as plt
from src.utils import compute_error
from src.metropolis_hastings import run_MH
from src.proj_langevin import run_PL, gen_init_point
from src.data_generation import generate_data
from src.utils import dump_run_information, dump_run_information_from_tensors


"""
Plot the accuracy of langevin vs prob wrt the rank of rho, with its rank being unknown.
This forces r=d for langevin. Here, n=4
"""
def run_experiment(savefig=True):
    seed = 0
    n = 4
    d = 2**n
    n_meas = d * d
    n_shots = 2000
    n_iter = 5000
    n_burnin = 1000
    eta_shots_indep = 0.005
    rho_ranks = range(1, d+1, 2)
    n_ranks = len(rho_ranks)
    n_samples = 8
    accs_prob = np.zeros((n_ranks, n_samples))
    accs_pl = np.zeros((n_ranks, n_samples))
    for i, rho_rank in enumerate(rho_ranks):
        for j in range(n_samples):
            curr_seed = seed + i*n_samples + j
            rho_true, As, y_hat = generate_data(n, n_meas, n_shots, rho_type=rho_rank, seed=curr_seed)
            # We don't know the rank of rho, hence we use r = d
            init_point = gen_init_point(d,d)
            As_flat = np.zeros((n_meas, 2**n * 2**n), dtype = np.complex128)
            for k in range(n_meas):
                As_flat[k,:] = As[:,:,k].flatten(order="C")
            _, rho_approx_prob, _ = run_MH(
                n, n_meas, n_shots, rho_true, As_flat, y_hat, n_iter, n_burnin, seed = None, init_point=init_point
            )
            _, rho_approx_pl, _ = run_PL(
                n, n_meas, n_shots, rho_rank, As, y_hat, n_iter, n_burnin, seed = None, init_point=init_point, eta_shots_indep=eta_shots_indep
            )
            
            err_prob = compute_error(rho_approx_prob, rho_true)
            err_pl = compute_error(rho_approx_pl, rho_true)

            accs_prob[i,j] = err_prob
            accs_pl[i,j] = err_pl
    accs_pl_avg = accs_pl.mean(axis=1)
    accs_prob_avg = accs_prob.mean(axis=1)
    plt.figure()
    plt.semilogy(rho_ranks, accs_pl_avg, "-o", label="langevin")
    plt.semilogy(rho_ranks, accs_prob_avg, "-o", label="prob")
    plt.legend()
    plt.xlabel(r"Rank of $\rho$ [#]")
    plt.ylabel("$L_2$ squared error")
    plt.title("Accuracy wrt rank, with burnin, rank of rho not known, n=4")
    if savefig:    
        plt.savefig(f"rank_not_known_n4.pdf", bbox_inches="tight")
    plt.show()
    plt.close()
    if savefig:
        dump_run_information_from_tensors(accs_prob, accs_pl, {"d": list(rho_ranks), "samples": list(range(n_samples))}, "rank_not_known_n4")
if __name__ == "__main__":
    run_experiment()
