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
Plots a phase transition, with n_shots/d on the x axis, and the rank of rho on y. Each point corresponds to the error for that combination.
Here, d is kept constant.
"""
def run_experiment(savefig=True):
    seed = 0
    n = 3
    d = 2**n
    n_meas = d*d
    rho_ranks = np.array(list(range(1, d+1)), dtype=int)
    shots = np.logspace(2, 7, 20, True, 10, dtype=np.int64)
    n_iter = 2000
    n_burnin = 500
    
    accs_prob = np.zeros((len(shots), len(rho_ranks)))
    accs_pl = np.zeros((len(shots), len(rho_ranks)))

    for i, n_shots in enumerate(shots):
        for j, rho_rank in enumerate(rho_ranks):
            rho_true, As, y_hat = generate_data(n, n_meas, n_shots, rho_type=rho_rank, seed=seed)
            As_flat = np.zeros((n_meas, 2**n * 2**n), dtype = np.complex128)
            for k in range(n_meas):
                As_flat[k,:] = As[:,:,k].flatten(order="C")
            _, rho_last_prob, _ = run_MH(n, n_meas, n_shots, rho_true, As_flat, y_hat, n_iter, n_burnin)
            _, rho_avg_pl, _  = run_PL(n, n_meas, n_shots, rho_rank, As, y_hat, n_iter, n_burnin)
            
            accs_prob[i,j] = np.log(compute_error(rho_last_prob, rho_true))
            accs_pl[i,j] = np.log(compute_error(rho_avg_pl, rho_true))
    
    
    xv, yv = np.meshgrid(shots/d, rho_ranks, indexing="ij")
    fig, axs = plt.subplots(1, 2)
    fig.suptitle("Phase transition of rank wrt n_shots/d")
    c1 = axs[0].contourf(xv, yv, accs_prob, cmap=plt.cm.rainbow,# type: ignore
                  vmin=accs_prob.min(), vmax=accs_prob.max())
    c2 = axs[1].contourf(xv, yv, accs_pl, cmap=plt.cm.rainbow,# type: ignore
                  vmin=accs_pl.min(), vmax=accs_pl.max())
    axs[0].set_title("prob")
    axs[0].set_xlabel("# of shots / d")
    axs[0].set_ylabel(r"Rank of $\rho$")

    axs[1].set_title("langevin")
    axs[1].set_xlabel("# of shots / d")

    
    plt.colorbar(mappable=c1, ax=axs[0])
    plt.colorbar(mappable=c2, ax=axs[1])

    if savefig:
        plt.savefig(f"phase_transition_shots.pdf", bbox_inches="tight")
    plt.show()

    dump_run_information("phase_transition_shots", {"shots": np.repeat(shots, len(rho_ranks)), "ranks": np.tile(rho_ranks, len(shots)), "acc_pl": accs_pl.flatten(), "acc_prob": accs_prob.flatten()})

    plt.show()
if __name__ == "__main__":
    run_experiment()
