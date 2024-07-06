import sys
sys.path.append("../../")

import numpy as np
import matplotlib.pyplot as plt
from src.utils import compute_error
from src.metropolis_hastings import run_MH
from src.proj_langevin import run_PL
from src.data_generation import generate_data
from src.utils import dump_run_information


"""
Plot the accuracy of langevin vs prob wrt to the number of iterations, 
no running average, number of qubits = 5
Note: for more accurate results with langevin, set the lr lower, even up to 10-4
"""
def run_experiment(savefig=True):
    seed = 0
    n = 5
    d = 2**n
    n_meas = d * d
    n_shots = 2000
    n_iter = 10000
    n_burnin = 2000
    rho_type = "rank2"

    rho_true, As, y_hat = generate_data(n, n_meas, n_shots, rho_type=rho_type, seed = seed)

    As_flat = np.zeros((n_meas, 2**n * 2**n), dtype = np.complex128)
    for i in range(n_meas):
        As_flat[i,:] = As[:,:,i].flatten(order="C")
    rhos_prob, _, cum_times_prob = run_MH(
        n, n_meas, n_shots, rho_true, As_flat, y_hat, n_iter, n_burnin
    )
    rhos_pl, _, cum_times_pl = run_PL(
        n, n_meas, n_shots, rho_type, As, y_hat, n_iter, n_burnin, eta_shots_indep=0.0001
    )

    accs_prob = [0.0] * (n_iter)
    accs_pl = [0.0] * (n_iter)
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
    plt.title("Accuracy wrt time, with burnin, n=5")
    if savefig:
        plt.savefig(f"iters_acc_comp_time_no_avg_n5.pdf", bbox_inches="tight")
    plt.show()
    plt.close()

    plt.figure()
    plt.semilogy(range(n_iter), accs_pl, label="langevin")
    plt.semilogy(range(n_iter), accs_prob, label="prob")
    plt.vlines(x=[n_burnin], ymin=0, ymax=0.1, color="r", label="end of burnin")
    plt.legend()
    plt.xlabel("Iterations [#]")
    plt.ylabel("$L_2$ squared error")
    plt.title("Accuracy wrt iters, with burnin, n=5")
    if savefig:    
        plt.savefig(f"iters_acc_comp_iters_no_avg_n5.pdf", bbox_inches="tight")
    plt.show()
    plt.close()

    dump_run_information("run_iterations_no_avg_n5", {"iter": list(range(n_iter)), "acc_pl": accs_pl, })

if __name__ == "__main__":
    run_experiment()
