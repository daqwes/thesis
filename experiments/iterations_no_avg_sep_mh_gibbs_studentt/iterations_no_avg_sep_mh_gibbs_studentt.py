import sys
sys.path.append("../../")

import numpy as np
import matplotlib.pyplot as plt
from src.utils import compute_error
from src.metropolis_hastings import run_MH
from src.proj_langevin import run_PL, gen_init_point
from src.mh_gibbs_studentt_prior import run_MH_gibbs_studentt
from src.data_generation_sep import generate_data_sep, generate_data_sep_PL 
from src.utils import dump_run_information, dump_run_information_from_tensors3


"""
Plot the accuracy of langevin vs prob vs mhgs with student-t wrt to the number of iterations, 
no running average for langevin, separate qubit DG
"""
def run_experiment(savefig=True):
    seed = 0
    n = 3
    d = 2**n
    n_meas = 3**n
    n_shots = 2000
    n_iter = 5000
    n_burnin = 1000
    rho_type = "rank2"

    rho_true, As, y_hat = generate_data_sep(n, n_meas, n_shots, rho_type=rho_type, seed=seed)
    _, As_PL, _ = generate_data_sep_PL(n, n_meas, n_shots, rho_type=rho_type, seed=seed)

    n_samples = 3
    accs_pl = np.zeros((n_iter, n_samples))
    accs_prob = np.zeros((n_iter, n_samples))
    accs_mhgs = np.zeros((n_iter, n_samples))

    for j in range(n_samples):
        # seed has be set above
        init_point = gen_init_point(d, d)
        # np.random.seed()
        rhos_prob, _, cum_times_prob = run_MH(
            n, n_meas, n_shots, rho_true, As, y_hat, n_iter, n_burnin, seed=None, init_point=init_point
        )
        rhos_pl, _, cum_times_pl = run_PL(
            n, n_meas, n_shots, rho_true, As_PL, y_hat, n_iter, n_burnin, seed=None, init_point=init_point
        )

        rhos_mhgs, _, cum_times_mh, acc_rate_mhgs = run_MH_gibbs_studentt(
            n, n_shots, As_PL, y_hat, n_iter, n_burnin, seed = None, proposal_dist="exp_dep", run_avg=True, scaling_coef_prop=0.1,  init_point=init_point
        )
 
        for i in range(n_iter):
            accs_prob[i,j] = compute_error(rhos_prob[i, :, :], rho_true)
            accs_pl[i,j] = compute_error(rhos_pl[i, :, :], rho_true)
            accs_mhgs[i,j] = compute_error(rhos_mhgs[i, :, :], rho_true)

    # plt.figure()
    # plt.semilogy(cum_times_pl[:n_burnin], accs_pl[:n_burnin])
    # plt.semilogy(cum_times_pl[n_burnin:], accs_pl[n_burnin:], label="langevin")

    # plt.semilogy(cum_times_prob[:n_burnin], accs_prob[:n_burnin])
    # plt.semilogy(cum_times_prob[n_burnin:], accs_prob[n_burnin:], label="prob")

    # plt.legend()
    # plt.xlabel("Time [s]")
    # plt.ylabel("$L_2$ squared error")
    # plt.title("Accuracy wrt time, with burnin, sep DG")
    # if savefig:
    #     plt.savefig(f"iters_acc_comp_time_no_avg_sep.pdf", bbox_inches="tight")
    # plt.show()
    # plt.close()

    avg_accs_pl = accs_pl.mean(axis=1)
    avg_accs_prob = accs_prob.mean(axis=1)
    avg_accs_mhgs = accs_mhgs.mean(axis=1)

    plt.figure()
    plt.semilogy(range(n_iter), avg_accs_pl, label="langevin")
    plt.semilogy(range(n_iter), avg_accs_prob, label="prob")
    plt.semilogy(range(n_iter), avg_accs_mhgs, label="mhgs")
    plt.vlines(x=[n_burnin], ymin=0, ymax=0.1, color="r", label="end of burnin")
    plt.legend()
    plt.xlabel("Iterations [#]")
    plt.ylabel("$L_2$ squared error")
    plt.title("Accuracy wrt iters, with burnin, sep DG, with mhgs")
    if savefig:    
        plt.savefig(f"iters_acc_comp_iters_no_avg_sep_mh_gibbs_studentt.pdf", bbox_inches="tight")
    plt.show()
    plt.close()
    if savefig:
        dump_run_information_from_tensors3(accs_prob, accs_pl, accs_mhgs, "mhgs", {"iter": list(range(n_iter)), "sample": list(range(n_samples))}, path="run_iterations_no_avg_sep_gibbs_mh_studentt")
        # dump_run_information("run_iterations_no_avg_sep_gibbs_mh_studentt", {"iter": list(range(n_iter)), "acc_pl": accs_pl, "acc_prob": accs_prob, "acc_mhs": accs_mhs})

if __name__ == "__main__":
    run_experiment()
