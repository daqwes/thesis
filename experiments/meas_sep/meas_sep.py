import sys
sys.path.append("../../")

import numpy as np
import matplotlib.pyplot as plt
from src.utils import compute_error
from src.metropolis_hastings import run_MH
from src.proj_langevin import run_PL, gen_init_point
from src.data_generation_sep import generate_data_sep, generate_data_sep_PL
from src.utils import dump_run_information, dump_run_information_from_tensors

"""
Plot the accuracy of langevin vs prob wrt to the number of measurements/observables (previously exp),
separate qubit DG
"""
def run_experiment(savefig=True):
    test_run = False
    # seed = 0
    n = 3
    d = 2**n
    n_meas = 3**n
    n_shots = 2000
    n_iter = 5000
    n_burnin = 1000
    rho_type = "rank2"
    meas = range(2, n_meas + 1, 2)
    n_samples = 6
    accs_prob = np.zeros((len(meas), n_samples))
    accs_pl = np.zeros((len(meas), n_samples))
    for j, n_meas in enumerate(meas):
        for k in range(n_samples):
            seed = k + n_samples * j
            rho_true, As, y_hat = generate_data_sep(n, n_meas, n_shots, rho_type=rho_type, seed = seed)
            _, As_PL, _ = generate_data_sep_PL(n, n_meas, n_shots, rho_type=rho_type, seed =seed)
            init_point = gen_init_point(d,d)
            if not test_run:
                _, rho_last_prob, _ = run_MH(n, n_meas, n_shots, rho_true, As, y_hat, n_iter, n_burnin, seed=None, init_point=init_point)
                _, rho_avg_pl, _  = run_PL(n, n_meas, n_shots, rho_type, As_PL, y_hat, n_iter, n_burnin, seed=None, init_point=init_point)
                
                accs_prob[j,k] = compute_error(rho_last_prob, rho_true)
                accs_pl[j,k] = compute_error(rho_avg_pl, rho_true)
            else:
                accs_prob[j,k] = seed
                accs_pl[j,k] = seed + 1

    avg_accs_prob = accs_prob.mean(axis=1)
    avg_accs_pl = accs_pl.mean(axis=1)
    plt.figure()
    plt.plot(meas, avg_accs_pl, label="langevin")
    plt.plot(meas, avg_accs_prob, label="prob")
    plt.legend()
    plt.xlabel("Number of measurements [#]")
    plt.ylabel("$L_2$ squared error")
    plt.title("Accuracy wrt n_meas (n_shots fixed), separate DG")
    if savefig:
         plt.savefig(f"meas_acc_comp_meas_sep.pdf", bbox_inches="tight")
    plt.show()

    plt.figure()
    plt.semilogy(meas, avg_accs_pl, label="langevin")
    plt.semilogy(meas, avg_accs_prob, label="prob")
    plt.legend()
    plt.xlabel("Number of measurements [#]")
    plt.ylabel("$L_2$ squared error")
    plt.title("Accuracy wrt n_meas, separate DG (n_shots fixed), semilogy")
    if savefig:
        plt.savefig(f"meas_acc_comp_meas_sep_semilogy.pdf", bbox_inches="tight")
    plt.show()

    if savefig:
        dump_run_information_from_tensors(accs_prob, accs_pl, {"n_meas": meas, "sample": list(range(n_samples))}, path="run_meas_sep")

if __name__ == "__main__":
    run_experiment()