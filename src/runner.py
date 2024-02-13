import numpy as np
from functools import partial
from typing import Callable, List
import time
import matplotlib.pyplot as plt
import pandas as pd

from data_generation import generate_data
from metropolis_hastings import run_MH
from proj_langevin import run_PL, random_complex_ortho

FIG_DIR  = "fig/"
DATA_DIR = "data/"
def time_run(f: Callable):
    """Times the execeution of a function
    Args:
        f (Callable): some function to time
    Returns:
        Tuple[float, Any] time and return values from function f 
    """
    tic = time.perf_counter()
    r_val = f()
    tac = time.perf_counter()
    return tac - tic, r_val 

def compute_error(rho_hat: np.ndarray, rho_true: np.ndarray, err_type: str = "fro_sq"):
    """
    """
    if err_type == "fro_sq":
        return np.linalg.norm(rho_hat - rho_true)**2
    elif err_type == "fro":
        return np.linalg.norm(rho_hat - rho_true)
    else:
        raise ValueError("No such error type")
    
def compare_iterations():

    n = 3
    d = 2**n
    n_exp = d * d
    n_shots = 2000
    
    # large = False
    # if large:
    #     iters = range(1000, 7000, 1000)
    #     ext = "_large"
    # else:
    #     iters = range(100, 2100, 100)
    #     ext = ""

    print(f"Iters compare")
    # large: 1000 - 6000, by 1000
    # normal: 100 - 2000, by 100
    rho_true, As, y_hat = generate_data(n, n_exp, n_shots, rho_type="rank2")
    accs_prob = []
    accs_pl = [] 
    n_iter = 10000
    n_burnin = 2000
    n_useful_iter = n_iter - n_burnin
    # for n_iter in iters:
    # all of them of size n_iter - n_burnin
    rhos_prob, rho_last, cum_times_prob = run_MH(n, n_exp, n_shots, rho_true, As, y_hat, n_iter, n_burnin)
    rhos_pl, rho_avg, cum_times_pl  = run_PL(n, n_exp, n_shots, rho_true, As, y_hat, n_iter, n_burnin)
    
    accs_prob = [0] * (n_iter)
    accs_pl = [0] * (n_iter)
    for i in range(n_iter):
        accs_prob[i] = compute_error(rhos_prob[i, :, :], rho_true)
        accs_pl[i] = compute_error(rhos_pl[i,:,:], rho_true)
    
    plt.figure()
    plt.semilogy(cum_times_pl[:n_burnin], accs_pl[:n_burnin])
    plt.semilogy(cum_times_pl[n_burnin:], accs_pl[n_burnin:], label="langevin")

    plt.semilogy(cum_times_prob[:n_burnin], accs_prob[:n_burnin])
    plt.semilogy(cum_times_prob[n_burnin:], accs_prob[n_burnin:], label="prob")

    # plt.plot(cum_times_pl, accs_pl, label="langevin")
    # plt.plot(cum_times_prob, accs_prob, label="prob")

    # ymax = max(max(accs_pl), max(accs_pl))
    # plt.vlines(x=[n_burnin], ymin=0, ymax=0.1, color="r", label="End of burnin")
    plt.legend()
    plt.xlabel("Time [s]")
    plt.ylabel("$L_2$ squared error")
    plt.title("Comparison of accuracy wrt time, with burnin")
    plt.savefig(FIG_DIR + f"iters_acc_comp_time.pdf", bbox_inches="tight")
    plt.show()
    plt.close()

    plt.figure()
    plt.semilogy(range(n_iter), accs_pl, label="langevin")
    plt.semilogy(range(n_iter), accs_prob, label="prob")
    plt.vlines(x=[n_burnin],ymin=0, ymax=0.1, color='r', label="end of burnin")
    plt.legend()
    plt.xlabel("Iterations [#]")
    plt.ylabel("$L_2$ squared error")
    plt.title("Comparison of accuracy wrt iters, with burnin")
    plt.savefig(FIG_DIR + f"iters_acc_comp_iters.pdf", bbox_inches="tight")
    plt.show()
    plt.close()

    # plt.figure()
    # plt.plot(iters, times_pl, label="langevin")
    # plt.plot(iters, times_prob, label="prob")
    # plt.legend()
    # plt.xlabel("Number of iterations [#]")
    # plt.ylabel("Time [s]")
    # plt.title("Comparison of timing wrt iterations")
    # plt.savefig(FIG_DIR + f"time_comp_iters{ext}.pdf", bbox_inches="tight")
    # plt.show()

    # plt.figure()
    # plt.plot(times_pl, accs_pl, label="langevin")
    # plt.plot(times_prob, accs_prob, label="prob")
    # plt.legend()
    # plt.xlabel("Time [s]")
    # plt.ylabel("$L_2$ squared error")
    # plt.title("Comparison of accuracy wrt time, increasing iters")
    # plt.savefig(FIG_DIR + f"iters_acc_comp_time.pdf", bbox_inches="tight")
    # plt.show()

    # n_iters = len(iters)
    # df = pd.DataFrame({"iter": np.tile(iters, 2), "acc": accs_pl + accs_prob, "times": times_pl + times_prob, "type": ["pl"] * n_iters + ["prob"] *  n_iters})
    # df.to_csv(DATA_DIR + f"iters{ext}.csv", ",", index=False)

def compare_nshots():
    n = 3
    d = 2**n
    n_exp = d * d


    shots_range = "large"
    if shots_range == "very_large":
        shots = range(1000, 21000, 1000)
        ext = "_verylarge"
    elif shots_range == "large":
        shots = range(500, 10500, 500)
        ext = "_large"
    else:
        shots = range(100, 2100, 100)
        ext = ""
    print(f"Shots compare {ext}")
    n_iter = 5000
    n_burnin = 1000
    accs_prob = []
    accs_pl = [] 
    for n_shots in shots:
        rho_true, As, y_hat = generate_data(n, n_exp, n_shots, rho_type="rank2")
        rhos_prob, rho_last_prob, cum_times_prob = run_MH(n, n_exp, n_shots, rho_true, As, y_hat, n_iter, n_burnin)
        rhos_pl, rho_avg_pl, cum_times_pl  = run_PL(n, n_exp, n_shots, rho_true, As, y_hat, n_iter, n_burnin)
        
        accs_prob.append(compute_error(rho_last_prob, rho_true))
        accs_pl.append(compute_error(rho_avg_pl, rho_true))

        # for i in range(n_iter):
        #     accs_prob[i] = compute_error(rhos_prob[i, :, :], rho_true)
        #     accs_pl[i] = compute_error(rhos_pl[i,:,:], rho_true)

    plt.figure()
    plt.semilogy(shots, accs_pl, label="langevin")
    plt.semilogy(shots, accs_prob, label="prob")
    plt.legend()
    plt.xlabel("Number of shots [#]")
    plt.ylabel("$L_2$ squared error")
    plt.title("Comparison of accuracy wrt shots")
    # plt.savefig(FIG_DIR + f"shots_acc_comp_shots{ext}.pdf", bbox_inches="tight")
    plt.show()

    # plt.figure()
    # plt.plot(times_pl, accs_pl, label="langevin")
    # plt.plot(times_prob, accs_prob, label="prob")
    # plt.legend()
    # plt.xlabel("Time [s]")
    # plt.ylabel("$L_2$ squared error")
    # plt.title("Comparison of accuracy wrt time, increasing shots")
    # plt.savefig(FIG_DIR + f"shots_acc_comp_time{ext}.pdf", bbox_inches="tight")
    # plt.show()

    # len_shots = len(shots)
    # df = pd.DataFrame({"shots": np.tile(shots, 2),"times": times_pl + times_prob, "acc": accs_pl + accs_prob, "type": ["pl"] * len_shots + ["prob"] *  len_shots})
    # df.to_csv(DATA_DIR + f"shots{ext}.csv", ",", index=False)

def compare_nexp():
    n = 3
    d = 2**n
    n_exp = d * d
    n_shots = 2000
    
    exps = range(2, d*d+1, 10) #TODO does not work for some reason

    print(f"Exps compare")
    n_iter = 5000
    n_burnin = 1000
    accs_prob = []
    accs_pl = [] 
    for n_exp in exps:
        rho_true, As, y_hat = generate_data(n, n_exp, n_shots, rho_type="rank2")
        rhos_prob, rho_last_prob, cum_times_prob = run_MH(n, n_exp, n_shots, rho_true, As, y_hat, n_iter, n_burnin)
        rhos_pl, rho_avg_pl, cum_times_pl  = run_PL(n, n_exp, n_shots, rho_true, As, y_hat, n_iter, n_burnin)
        
        accs_prob.append(compute_error(rho_last_prob, rho_true))
        accs_pl.append(compute_error(rho_avg_pl, rho_true))

    plt.figure()
    plt.plot(exps, accs_pl, label="langevin")
    plt.plot(exps, accs_prob, label="prob")
    plt.legend()
    plt.xlabel("Number of experiments [#]")
    plt.ylabel("$L_2$ squared error")
    plt.title("Comparison of accuracy wrt n_exp")
    plt.savefig(FIG_DIR + f"exps_acc_comp_exps.pdf", bbox_inches="tight")
    plt.show()


def run():

    n = 3
    d = 2**n
    n_exp = d * d
    n_shots = 2000
    rho_true, As, y_hat = generate_data(n, n_exp, n_shots, rho_type="rank2")

    prob = partial(run_MH, n, n_shots, rho_true, As, y_hat)
    pl = partial(run_PL, n, n_exp, n_shots, rho_true, As, y_hat)

    time_prob, rho_prob = time_run(prob)
    time_pl, rho_pl = time_run(pl)

    err_prob = compute_error(rho_prob, rho_true)
    err_pl = compute_error(rho_pl, rho_true)
    
    print(f"Prob-estimator: {time_prob}s - {err_prob:.3e}")
    print(f"Proj-langevin: {time_pl}s - {err_pl:.3e}")


if __name__ == "__main__":
    # seed = 0
    # np.random.seed(seed)
    compare_nshots()