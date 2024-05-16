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
Plots a phase transition, with n_meas/d on the x axis, and the rank of rho on y. Each point corresponds to the error for that combination.
Here, d is kept constant.
"""
def run_experiment(savefig=True):
    seed = 0
    n = 3
    d = 2**n
    n_exps = np.array(list(range(1, d*d + 1, 5)), dtype=int)
    rho_ranks = np.array(list(range(1, d+1)), dtype=int)
    n_shots = 2000
    n_iter = 2000
    n_burnin = 500
    
    accs_prob = np.zeros((len(n_exps), len(rho_ranks)))
    accs_pl = np.zeros((len(n_exps), len(rho_ranks)))

    for i, n_meas in enumerate(n_exps):
        for j, rho_rank in enumerate(rho_ranks):
            rho_true, As, y_hat = generate_data(n, n_meas, n_shots, rho_type=rho_rank, seed=seed)
            As_flat = np.zeros((n_meas, 2**n * 2**n), dtype = np.complex128)
            for k in range(n_meas):
                # TODO: it is not clear why this works better than `flatten(order="F")`
                # as it is more correct to use the latter (similar to what is done in R)
                As_flat[k,:] = As[:,:,k].flatten(order="C")
            _, rho_last_prob, _ = run_MH(n, n_meas, n_shots, rho_true, As_flat, y_hat, n_iter, n_burnin)
            _, rho_avg_pl, _  = run_PL(n, n_meas, n_shots, rho_rank, As, y_hat, n_iter, n_burnin)
            
            accs_prob[i,j] = np.log(compute_error(rho_last_prob, rho_true))
            accs_pl[i,j] = np.log(compute_error(rho_avg_pl, rho_true))
    
    
    xv, yv = np.meshgrid(n_exps/d, rho_ranks, indexing="ij")
    fig, axs = plt.subplots(1, 2)
    fig.suptitle("Phase transition of rank wrt n_meas/d")
    c1 = axs[0].contourf(xv, yv, accs_prob, cmap=plt.cm.rainbow,# type: ignore
                  vmin=accs_prob.min(), vmax=accs_prob.max())
    c2 = axs[1].contourf(xv, yv, accs_pl, cmap=plt.cm.rainbow,# type: ignore
                  vmin=accs_pl.min(), vmax=accs_pl.max())
    axs[0].set_title("prob")
    axs[0].set_xlabel("# of measurements / d")
    axs[0].set_ylabel(r"Rank of $\rho$")

    axs[1].set_title("langevin")
    axs[1].set_xlabel("# of measurements / d")

    
    plt.colorbar(mappable=c1, ax=axs[0])
    plt.colorbar(mappable=c2, ax=axs[1])

    if savefig:
        plt.savefig(f"phase_transition_exps.pdf", bbox_inches="tight")
    plt.show()

    if savefig:
        dump_run_information("phase_transition_exps", {"exps": np.repeat(n_exps, len(rho_ranks)), "ranks": np.tile(rho_ranks, len(n_exps)), "acc_pl": accs_pl.flatten(), "acc_prob": accs_prob.flatten()})

    plt.show()
if __name__ == "__main__":
    run_experiment()
