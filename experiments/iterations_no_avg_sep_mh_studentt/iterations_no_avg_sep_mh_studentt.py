import sys
sys.path.append("../../")

import numpy as np
import matplotlib.pyplot as plt
from src.utils import compute_error
from src.metropolis_hastings import run_MH
from src.proj_langevin import run_PL, gen_init_point
from src.mh_studentt_prior import run_MH_studentt
from src.data_generation_sep import generate_data_sep, generate_data_sep_PL 
from src.utils import dump_run_information


"""
Plot the accuracy of langevin vs prob vs mh with student-t wrt to the number of iterations, 
no running average for langevin, separate qubit DG
"""
def run_experiment(savefig=True):
    seed = 0
    n = 3
    d = 2**n
    n_meas = 3**n
    n_shots = 2000
    n_iter = 10000
    n_burnin = 2000
    rho_type = "rank2"

    rho_true, As, y_hat = generate_data_sep(n, n_meas, n_shots, rho_type=rho_type, seed=seed)
    _, As_PL, _ = generate_data_sep_PL(n, n_meas, n_shots, rho_type=rho_type, seed=seed)

    # seed has be set above
    init_point = gen_init_point(d, d)
    # np.random.seed()
    rhos_prob, _, cum_times_prob = run_MH(
        n, n_meas, n_shots, rho_true, As, y_hat, n_iter, n_burnin, seed=None, init_point=init_point
    )
    rhos_pl, _, cum_times_pl = run_PL(
        n, n_meas, n_shots, rho_type, As_PL, y_hat, n_iter, n_burnin, seed=None, init_point=init_point
    )

    rhos_mhs, _, cum_times_mh, acc_rate_mhs = run_MH_studentt(
        n, n_shots, As_PL, y_hat, n_iter, n_burnin, seed = None, proposal_dist="exp_dep", run_avg=True, scaling_coef_prop=0.1,  init_point=init_point
    )

    accs_prob = [0.0] * (n_iter)
    accs_pl = [0.0] * (n_iter)
    accs_mhs = [0.0] * n_iter
    for i in range(n_iter):
        accs_prob[i] = compute_error(rhos_prob[i, :, :], rho_true)
        accs_pl[i] = compute_error(rhos_pl[i, :, :], rho_true)
        accs_mhs[i] = compute_error(rhos_mhs[i, :, :], rho_true)
        
    plt.figure()
    plt.semilogy(range(n_iter), accs_pl, label="langevin")
    plt.semilogy(range(n_iter), accs_prob, label="prob")
    plt.semilogy(range(n_iter), accs_mhs, label="mhs")
    plt.vlines(x=[n_burnin], ymin=0, ymax=0.1, color="r", label="end of burnin")
    plt.legend()
    plt.xlabel("Iterations [#]")
    plt.ylabel("$L_2$ squared error")
    plt.title("Accuracy wrt iters, with burnin, sep DG, with mhs")
    if savefig:    
        plt.savefig(f"iters_acc_comp_iters_no_avg_sep_mh_studentt.pdf", bbox_inches="tight")
    plt.show()
    plt.close()
    if savefig:
        dump_run_information("run_iterations_no_avg_sep_mh_studentt", {"iter": list(range(n_iter)), "acc_pl": accs_pl, "acc_prob": accs_prob, "acc_mhs": accs_mhs})

if __name__ == "__main__":
    run_experiment()
