import sys
sys.path.append("../../")

import numpy as np
import matplotlib.pyplot as plt
from src.utils import compute_error
from src.metropolis_hastings import run_MH
from src.proj_langevin import run_PL
from src.data_generation_sep import generate_data_sep, generate_data_sep_PL
from src.utils import dump_run_information

"""
Evolution of the accuracy wrt the number of shots, separate qubit DG.
"""
def run_experiment(savefig=True):
    
    n = 3
    d = 2**n
    n_meas = 3**n
    n_iter = 5000
    n_burnin = 1000
    rho_type = "rank2"
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

    accs_prob = []
    accs_pl = [] 
    for n_shots in shots:
        seed = 0
        rho_true, As, y_hat = generate_data_sep(n, n_meas, n_shots, rho_type=rho_type, seed=seed)
        _, As_PL, _ = generate_data_sep_PL(n, n_meas, n_shots, rho_type=rho_type, seed=seed)
        _, rho_last_prob, _ = run_MH(n, n_meas, n_shots, rho_true, As, y_hat, n_iter, n_burnin, seed=None)
        _, rho_avg_pl, _  = run_PL(n, n_meas, n_shots, rho_type, As_PL, y_hat, n_iter, n_burnin, seed=None)
        
        accs_prob.append(compute_error(rho_last_prob, rho_true))
        accs_pl.append(compute_error(rho_avg_pl, rho_true))

    plt.figure()
    plt.semilogy(shots, accs_pl, label="langevin")
    plt.semilogy(shots, accs_prob, label="prob")
    plt.legend()
    plt.xlabel("Number of shots [#]")
    plt.ylabel("$L_2$ squared error")
    plt.title("Accuracy wrt shots, sep DG, semilogy")
    plt.savefig(f"shots_acc_comp_shots{ext}_sep.pdf", bbox_inches="tight")
    plt.show()

    plt.figure()
    plt.loglog(shots, accs_pl, label="langevin")
    plt.loglog(shots, accs_prob, label="prob")
    plt.legend()
    plt.xlabel("Number of shots [#]")
    plt.ylabel("$L_2$ squared error")
    plt.title("Accuracy wrt shots, sep DG, loglog")
    if savefig:
        plt.savefig(f"shots_acc_comp_shots{ext}_sep_loglog.pdf", bbox_inches="tight")
    plt.show()

    dump_run_information("run_shots_sep", {"shots": shots, "acc_pl": accs_pl, "acc_prob": accs_prob})  



if __name__ == "__main__":
    run_experiment()