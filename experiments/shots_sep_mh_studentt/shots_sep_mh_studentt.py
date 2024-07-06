import sys
sys.path.append("../../")

import numpy as np
import matplotlib.pyplot as plt
from src.utils import compute_error
from src.metropolis_hastings import run_MH
from src.proj_langevin import run_PL, gen_init_point
from src.data_generation_sep import generate_data_sep, generate_data_sep_PL
from src.utils import dump_run_information
from src.mh_studentt_prior import run_MH_studentt

"""
Evolution of the accuracy wrt the number of shots, separate qubit DG, with mhs.
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
    accs_mhs = [] 

    for n_shots in shots:
        seed = 0
        rho_true, As, y_hat = generate_data_sep(n, n_meas, n_shots, rho_type=rho_type, seed=seed)
        _, As_PL, _ = generate_data_sep_PL(n, n_meas, n_shots, rho_type=rho_type, seed=seed)
        init_point = gen_init_point(d,d)
        _, rho_last_prob, _ = run_MH(n, n_meas, n_shots, rho_true, As, y_hat, n_iter, n_burnin, seed=None, init_point=init_point)
        _, rho_avg_pl, _  = run_PL(n, n_meas, n_shots, rho_type, As_PL, y_hat, n_iter, n_burnin, seed=None, init_point=init_point)
        _, rho_avg_mhs, _, _ = run_MH_studentt(
            n, n_shots, As_PL, y_hat, n_iter, n_burnin, seed = None, proposal_dist="exp_dep", run_avg=True, scaling_coef_prop=0.1,  init_point=init_point
        )

        accs_prob.append(compute_error(rho_last_prob, rho_true))
        accs_pl.append(compute_error(rho_avg_pl, rho_true))
        accs_mhs.append(compute_error(rho_avg_mhs, rho_true))

    plt.figure()
    plt.semilogy(shots, accs_pl, label="langevin")
    plt.semilogy(shots, accs_prob, label="prob")
    plt.semilogy(shots, accs_mhs, label="mhs")

    plt.legend()
    plt.xlabel("Number of shots [#]")
    plt.ylabel("$L_2$ squared error")
    plt.title("Accuracy wrt shots, sep DG, semilogy, with mhs")
    plt.savefig(f"shots_acc_comp_shots{ext}_sep_mh_studentt.pdf", bbox_inches="tight")
    plt.show()

    plt.figure()
    plt.loglog(shots, accs_pl, label="langevin")
    plt.loglog(shots, accs_prob, label="prob")
    plt.loglog(shots, accs_mhs, label="prob")
    plt.legend()
    plt.xlabel("Number of shots [#]")
    plt.ylabel("$L_2$ squared error")
    plt.title("Accuracy wrt shots, sep DG, loglog, with mhs")
    if savefig:
        plt.savefig(f"shots_acc_comp_shots{ext}_sep_mhs_loglog.pdf", bbox_inches="tight")
    plt.show()

    if savefig:
        dump_run_information("run_shots_sep_mh_studentt", {"shots": shots, "acc_pl": accs_pl, "acc_prob": accs_prob, "acc_mhs": accs_mhs})  

if __name__ == "__main__":
    run_experiment()