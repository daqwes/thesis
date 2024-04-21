import sys
sys.path.append("../../")

import numpy as np
import matplotlib.pyplot as plt
from src.utils import compute_error
from src.metropolis_hastings import run_MH
from src.proj_langevin import run_PL, gen_init_point
from src.data_generation_exact import generate_data_exact, generate_data_exact_PL
from src.utils import dump_run_information

"""
Compare the accuracy of the methods given different burnin periods. No running average for langevin. Exact data generation.
"""
def run_experiment(savefig=True):
    n = 3
    d = 2**n
    n_meas = 3**n
    n_iter = 10000
    n_shots = 2000
    rho_type="rank2"

    burnin_range = range(100, 6000, 300)

    accs_prob = []
    accs_pl = [] 
    seed = 0
    rho_true, As, y_hat = generate_data_exact(n, n_meas, n_shots, rho_type=rho_type, seed=seed)
    _, As_PL, _ = generate_data_exact_PL(n, n_meas, n_shots, rho_type=rho_type, seed=seed)
    init_point = gen_init_point(d, d)
    # np.random.seed(seed + 1)
    for n_burnin in burnin_range:
        np.random.seed(seed + 1)
        _, rho_last_prob, _ = run_MH(n, n_meas, n_shots, rho_true, As, y_hat, n_iter, n_burnin, seed=None, init_point=init_point)
        _, rho_avg_pl, _  = run_PL(n, n_meas, n_shots, rho_true, As_PL, y_hat, n_iter, n_burnin, seed=None, init_point=init_point)
        
        accs_prob.append(compute_error(rho_last_prob, rho_true))
        accs_pl.append(compute_error(rho_avg_pl, rho_true))

    plt.figure()
    plt.semilogy(burnin_range, accs_pl, label="langevin")
    plt.semilogy(burnin_range, accs_prob, label="prob")
    plt.legend()
    plt.xlabel("Length of burnin period [#]")
    plt.ylabel("$L_2$ squared error")
    plt.title("Comparison of accuracy wrt burnin length, exact, semilogy")
    if savefig:    
        plt.savefig(f"burnin_acc_comp_burnin_exact.pdf", bbox_inches="tight")
    plt.show()

    # plt.figure()
    # plt.loglog(shots, accs_pl, label="langevin")
    # plt.loglog(shots, accs_prob, label="prob")
    # plt.legend()
    # plt.xlabel("Number of shots [#]")
    # plt.ylabel("$L_2$ squared error")
    # plt.title("Comparison of accuracy wrt shots, loglog")
    # if savefig:
    #     plt.savefig(f"shots_acc_comp_shots{ext}_loglog.pdf", bbox_inches="tight")
    # plt.show()

    if savefig:
        dump_run_information("run_burnin_no_avg_exact", {"n_burnin": burnin_range, "acc_pl": accs_pl, "acc_prob": accs_prob})  


if __name__ == "__main__":
    run_experiment()