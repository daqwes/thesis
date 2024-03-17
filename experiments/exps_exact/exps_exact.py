import sys
sys.path.append("../../")

import numpy as np
import matplotlib.pyplot as plt
from src.utils import compute_error
from src.metropolis_hastings import run_MH
from src.proj_langevin import run_PL
from src.data_generation_exact import generate_data_exact, generate_data_exact_PL
from src.utils import dump_run_information


"""
Plot the accuracy of langevin vs prob wrt to the number of experiments
exact data generation
"""
def run_experiment(savefig=True):
    seed = 0
    n = 3
    d = 2**n
    n_exp = 3**n
    n_shots = 2000
    rho_type = "rank2"
    exps = range(2, n_exp + 1, 10)
    n_iter = 5000
    n_burnin = 1000
    
    print(f"Exps compare, exact")
    accs_prob = []
    accs_pl = [] 
    for n_exp in exps:
        rho_true, As, y_hat = generate_data_exact(n, n_exp, n_shots, rho_type=rho_type, seed = seed)
        _, As_PL, _ = generate_data_exact_PL(n, n_exp, n_shots, rho_type=rho_type, seed =seed)

        _, rho_last_prob, _ = run_MH(n, n_exp, n_shots, rho_true, As, y_hat, n_iter, n_burnin)
        _, rho_avg_pl, _  = run_PL(n, n_exp, n_shots, rho_true, As_PL, y_hat, n_iter, n_burnin)
        
        accs_prob.append(compute_error(rho_last_prob, rho_true))
        accs_pl.append(compute_error(rho_avg_pl, rho_true))

    plt.figure()
    plt.semilogy(exps, accs_pl, label="langevin")
    plt.semilogy(exps, accs_prob, label="prob")
    plt.legend()
    plt.xlabel("Number of experiments [#]")
    plt.ylabel("$L_2$ squared error")
    plt.title("Accuracy wrt n_exp, exact data")
    if savefig:
        plt.savefig(f"exps_acc_comp_exps_exact.pdf", bbox_inches="tight")
    plt.show()

    dump_run_information("run_exps_exact", {"exps": exps, "acc_pl": accs_pl, "acc_prob": accs_prob})



if __name__ == "__main__":
    run_experiment()