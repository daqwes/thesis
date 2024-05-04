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
Evolution of the accuracy wrt the number of shots
"""
def run_experiment(savefig=True):
    n = 3
    d = 2**n
    n_meas = d * d
    n_iter = 5000
    n_burnin = 1000
    rho_type="rank2"
    eta_shots_indep = 0.005
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

    n_samples = 5 
    accs_prob = np.zeros((len(shots), n_samples))
    accs_pl = np.zeros((len(shots), n_samples))
    for i, n_shots in enumerate(shots):
        for j in range(n_samples):
            seed = j + n_samples * i
            rho_true, As, y_hat = generate_data(n, n_meas, n_shots, rho_type=rho_type, seed = seed)
            As_flat = np.zeros((n_meas, 2**n * 2**n), dtype = np.complex128)
            for k in range(n_meas):
                # TODO: it is not clear why this works better than `flatten(order="F")`
                # as it is more correct to use the latter (similar to what is done in R)
                As_flat[k,:] = As[:,:,k].flatten(order="C")
            init_point = gen_init_point(d,d)
            _, rho_last_prob, _ = run_MH(n, n_meas, n_shots, rho_true, As_flat, y_hat, n_iter, n_burnin, seed=None, init_point=init_point)
            _, rho_avg_pl, _  = run_PL(n, n_meas, n_shots, rho_true, As, y_hat, n_iter, n_burnin, seed=None, init_point=init_point, eta_shots_indep=eta_shots_indep)
            
            accs_prob[i,j] = compute_error(rho_last_prob, rho_true)
            accs_pl[i,j] = compute_error(rho_avg_pl, rho_true)

    avg_accs_prob = accs_prob.mean(axis=1)
    avg_accs_pl = accs_pl.mean(axis=1)


    plt.figure()
    plt.semilogy(shots, avg_accs_pl, label="langevin")
    plt.semilogy(shots, avg_accs_prob, label="prob")
    plt.legend()
    plt.xlabel("Number of shots [#]")
    plt.ylabel("$L_2$ squared error")
    plt.title("Comparison of accuracy wrt shots, semilogy")
    if savefig:    
        plt.savefig(f"shots_acc_comp_shots{ext}.pdf", bbox_inches="tight")
    plt.show()

    plt.figure()
    plt.loglog(shots, avg_accs_pl, label="langevin")
    plt.loglog(shots, avg_accs_prob, label="prob")
    plt.legend()
    plt.xlabel("Number of shots [#]")
    plt.ylabel("$L_2$ squared error")
    plt.title("Comparison of accuracy wrt shots, loglog")
    if savefig:
        plt.savefig(f"shots_acc_comp_shots{ext}_loglog.pdf", bbox_inches="tight")
    plt.show()

    if savefig:
        dump_run_information_from_tensors(accs_prob, accs_pl, {"shots": shots, "sample": list(range(n_samples))}, "run_shots")
    # dump_run_information("run_shots", {"shots": shots, "acc_pl": accs_pl, "acc_prob": accs_prob})  

if __name__ == "__main__":
    run_experiment()