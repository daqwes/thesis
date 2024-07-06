import sys
import pickle
sys.path.append("../../")

import numpy as np
import matplotlib.pyplot as plt
from src.utils import compute_error
from src.metropolis_hastings import run_MH
from src.proj_langevin import run_PL, gen_init_point
from src.data_generation import generate_data
from src.utils import dump_run_information, dump_run_information_from_tensors
from src.plotting import lighten_color

"""
Plot the accuracy of langevin vs prob wrt the rank of rho, with its rank assumed known.
This means that r = rank(rho) for langevin. Here, we want to see how the result changes when theta changes for langevin.

"""
def run_experiment(savefig=True):
    test_run = False
    savefig = not test_run
    seed = 0
    n = 3
    d = 2**n
    n_meas = d * d
    n_shots = 2000
    n_iter = 5000
    n_burnin = 1000
    eta_shots_indep = 0.005
    thetas_pl = [1e-6,  1e-3, 1e-2, 1e-1, 1e0, 1e1]
    rho_ranks = range(1, d+1)
    n_samples = 10
    accs_prob = np.zeros((len(thetas_pl), d, n_samples))
    accs_pl = np.zeros((len(thetas_pl), d, n_samples))
    saved_res_prob = np.zeros((d, n_samples))
    for i, theta_pl in enumerate(thetas_pl):            
        for j, rho_rank in enumerate(rho_ranks):
            for k in range(n_samples):
                # k,i,j -> j + n_samples * i + len(shots) * n_samples * k
                # i,j,k -> 
                # curr_seed = seed + i*n_samples + j
                if theta_pl < 1e-3:
                    eta_shots_indep = 0.0005
                else:
                    eta_shots_indep = 0.005
                curr_seed = seed + k + n_samples * j + n_samples * len(rho_ranks) * i
                print(curr_seed)
                rho_true, As, y_hat = generate_data(n, n_meas, n_shots, rho_type=rho_rank, seed=curr_seed)
                init_point_MH = gen_init_point(d, d)
                # We know the rank of rho, hence we use r = rank(rho)
                init_point_PL = gen_init_point(d, rho_rank)
                As_flat = np.zeros((n_meas, 2**n * 2**n), dtype = np.complex128)
                for l in range(n_meas):
                    As_flat[l,:] = As[:,:,l].flatten(order="C")
                if i == 0:
                    if not test_run:
                        _, rho_approx_prob, _ = run_MH(
                            n, n_meas, n_shots, rho_true, As_flat, y_hat, n_iter, n_burnin, seed=None, init_point=init_point_MH
                        )
                        err_prob = compute_error(rho_approx_prob, rho_true)
                    else:
                        err_prob = curr_seed + 1
                    saved_res_prob[j,k] = err_prob
                else:
                    err_prob = saved_res_prob[j,k]
                if not test_run:
                    _, rho_approx_pl, _ = run_PL(
                        n, n_meas, n_shots, rho_rank, As, y_hat, n_iter, n_burnin, seed=None, init_point=init_point_PL, eta_shots_indep=eta_shots_indep, theta=theta_pl
                    )
                    err_pl = compute_error(rho_approx_pl, rho_true)
                else:
                    err_pl = curr_seed + 2
                accs_prob[i,j,k] = err_prob
                accs_pl[i,j,k] = err_pl

    accs_pl_avg = accs_pl.mean(axis=2)
    accs_prob_avg = accs_prob.mean(axis=2)
    plt.figure()
    plt.grid()
    plt.semilogy(rho_ranks, accs_prob_avg[0], "-o", label="prob", c="blue")
    main_color_pl = "orange"
    colors_prob = [lighten_color(main_color_pl, amount=s) for s in np.linspace(1.6, 0.2, len(thetas_pl))]
    for i in range(len(thetas_pl)):
        plt.semilogy(rho_ranks, accs_pl_avg[i], "-o", label=rf"langevin,$\theta$={thetas_pl[i]}", c=colors_prob[i])

    plt.legend()
    plt.xlabel(r"Rank of $\rho$ [#]")
    plt.ylabel("$L_2$ squared error")
    plt.title("Accuracy wrt rank, with burnin, rank of rho known, var theta for PL")
    if savefig:    
        plt.savefig(f"rank_known_var_theta_pl.pdf", bbox_inches="tight")
    plt.show()
    plt.close()
    if savefig:
        filename = "rank_known_var_theta_pl"
        params = {"theta": thetas_pl, "d": list(rho_ranks), "samples": list(range(n_samples))}
        with open("accs_prob_" + filename + ".pkl", "wb") as f:
            pickle.dump(accs_prob, f)
        with open("accs_pl_" + filename + ".pkl", "wb") as f:
            pickle.dump(accs_pl, f)
        with open("params_" + filename + ".pkl", "wb") as f:
            pickle.dump(params, f)
        dump_run_information_from_tensors(accs_prob, accs_pl, params, filename)
if __name__ == "__main__":
    run_experiment()
