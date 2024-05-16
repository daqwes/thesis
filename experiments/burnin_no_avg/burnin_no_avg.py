import sys
sys.path.append("../../")

import numpy as np
import matplotlib.pyplot as plt
from src.utils import compute_error
from src.metropolis_hastings import run_MH
from src.proj_langevin import run_PL, gen_init_point
from src.data_generation import generate_data
from src.utils import dump_run_information

"""
Compare the accuracy of the methods given different burnin periods. No running average for langevin
"""
def run_experiment(savefig=True):
    n = 3
    d = 2**n
    n_meas = d * d
    n_iter = 10000
    n_shots = 2000
    rho_type="rank2"

    burnin_range = range(100, 6000, 300)

    accs_prob = []
    accs_pl = []
    seed = 0
    rho_true, As, y_hat = generate_data(n, n_meas, n_shots, rho_type=rho_type, seed = seed)
    As_flat = np.zeros((n_meas, 2**n * 2**n), dtype = np.complex128)
    for i in range(n_meas):
        # TODO: it is not clear why this works better than `flatten(order="F")`
        # as it is more correct to use the latter (similar to what is done in R)
        As_flat[i,:] = As[:,:,i].flatten(order="C")
    init_point = gen_init_point(d, d)
    for n_burnin in burnin_range:
        np.random.seed(seed + 1)
        _, rho_last_prob, _ = run_MH(n, n_meas, n_shots, rho_true, As_flat, y_hat, n_iter, n_burnin, seed=None, init_point=init_point)
        _, rho_avg_pl, _  = run_PL(n, n_meas, n_shots, rho_type, As, y_hat, n_iter, n_burnin, seed=None, init_point=init_point)
        
        accs_prob.append(compute_error(rho_last_prob, rho_true))
        accs_pl.append(compute_error(rho_avg_pl, rho_true))

    plt.figure()
    plt.semilogy(burnin_range, accs_pl, label="langevin")
    plt.semilogy(burnin_range, accs_prob, label="prob")
    plt.legend()
    plt.xlabel("Length of burnin period [#]")
    plt.ylabel("$L_2$ squared error")
    plt.title("Comparison of accuracy wrt burnin length, semilogy")
    if savefig:    
        plt.savefig(f"burnin_acc_comp_burnin.pdf", bbox_inches="tight")
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
        dump_run_information("run_burnin_no_avg", {"n_burnin": burnin_range, "acc_pl": accs_pl, "acc_prob": accs_prob})  


if __name__ == "__main__":
    run_experiment()