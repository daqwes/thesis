import sys
sys.path.append("../../")

import numpy as np
import matplotlib.pyplot as plt
from src.utils import compute_error
from src.metropolis_hastings import run_MH
from src.proj_langevin import run_PL
from src.data_generation_sep import generate_data_exact, generate_data_exact_PL 
from src.utils import dump_run_information


"""
Plot the accuracy of langevin vs prob wrt to the number of iterations, 
no running average, separate qubit DG, number of qubits = 4
Note: for more accurate results with langevin, set the lr lower, even up to 10-4
"""
def run_experiment(savefig=True):
    seed = 0
    n = 4
    d = 2**n
    n_meas = 3**n
    n_shots = 2000
    n_iter = 10000
    n_burnin = 2000
    rho_type = "rank2"

    rho_true, As, y_hat = generate_data_exact(n, n_meas, n_shots, rho_type=rho_type, seed=seed)
    _, As_PL, _ = generate_data_exact_PL(n, n_meas, n_shots, rho_type=rho_type, seed=seed)
    accs_prob = []
    accs_pl = []

    rhos_prob, _, cum_times_prob = run_MH(
        n, n_meas, n_shots, rho_true, As, y_hat, n_iter, n_burnin, seed = None
    )
    rhos_pl, _, cum_times_pl = run_PL(
        n, n_meas, n_shots, rho_true, As_PL, y_hat, n_iter, n_burnin, seed=None
    )

    accs_prob = [0] * (n_iter)
    accs_pl = [0] * (n_iter)
    for i in range(n_iter):
        accs_prob[i] = compute_error(rhos_prob[i, :, :], rho_true)
        accs_pl[i] = compute_error(rhos_pl[i, :, :], rho_true)

    plt.figure()
    plt.semilogy(cum_times_pl[:n_burnin], accs_pl[:n_burnin])
    plt.semilogy(cum_times_pl[n_burnin:], accs_pl[n_burnin:], label="langevin")

    plt.semilogy(cum_times_prob[:n_burnin], accs_prob[:n_burnin])
    plt.semilogy(cum_times_prob[n_burnin:], accs_prob[n_burnin:], label="prob")

    plt.legend()
    plt.xlabel("Time [s]")
    plt.ylabel("$L_2$ squared error")
    plt.title("Accuracy wrt time, with burnin, sep DG, n=4")
    if savefig:
        plt.savefig(f"iters_acc_comp_time_no_avg_exact_n4.pdf", bbox_inches="tight")
    plt.show()
    plt.close()

    plt.figure()
    plt.semilogy(range(n_iter), accs_pl, label="langevin")
    plt.semilogy(range(n_iter), accs_prob, label="prob")
    plt.vlines(x=[n_burnin], ymin=0, ymax=0.1, color="r", label="end of burnin")
    plt.legend()
    plt.xlabel("Iterations [#]")
    plt.ylabel("$L_2$ squared error")
    plt.title("Accuracy wrt iters, with burnin, sep DG, n=4")
    if savefig:    
        plt.savefig(f"iters_acc_comp_iters_no_avg_exact_n4.pdf", bbox_inches="tight")
    plt.show()
    plt.close()

    dump_run_information("run_iterations_no_avg_exact_n4", {"iter": list(range(n_iter)), "acc_pl": accs_pl, "acc_prob": accs_prob})

if __name__ == "__main__":
    run_experiment()
