import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
sys.path.append("../../")
from src.plotting import lighten_color

def read_res_var_shots():
    filename = "../shots/run_shots.csv"
    df = pd.read_csv(filename)
    shots = np.logspace(2, 7, 20, True, 10, dtype=np.int64)
    rev_map_shots = {v: i for i, v in enumerate(shots)}
    n_samples = 5
    accs_prob_var_shots = np.zeros((len(shots), n_samples))

    for _, data in df.iterrows():
        idx_shot = rev_map_shots[int(data.shots)]
        idx_sample = int(data["sample"])
        accs_prob_var_shots[idx_shot, idx_sample] = data["acc_prob"]

    return accs_prob_var_shots

def main(savefig=True):
    filename = "../shots_lambda_prob/run_shots_lambda_prob.csv"
    df = pd.read_csv(filename)

    lambdas = range(500, 5500, 500) # 10 values
    rev_map_lambdas = {v: i for i, v in enumerate(lambdas)}
    shots = np.logspace(2, 7, 20, True, 10, dtype=np.int64)
    rev_map_shots = {v: i for i, v in enumerate(shots)}

    ext = "_exp"
    n_samples = 3

    accs_pl = np.zeros((len(lambdas), len(shots), n_samples))
    accs_prob = np.zeros((len(lambdas), len(shots), n_samples))

    for _, data in df.iterrows():
        # Header: lambda,shots,sample,acc_prob,acc_pl
        # print(data)
        # print(data["lambda"])
        # exit(0)
        idx_lambda = rev_map_lambdas[int(data["lambda"])]
        idx_shot = rev_map_shots[int(data.shots)]
        idx_sample = int(data["sample"])
        accs_pl[idx_lambda, idx_shot, idx_sample] = data["acc_pl"]
        accs_prob[idx_lambda, idx_shot, idx_sample] = data["acc_prob"]


    accs_prob_var_shots = read_res_var_shots()

    avg_accs_prob = accs_prob.mean(axis=2)
    avg_accs_pl = accs_pl.mean(axis=2)
    avg_accs_prob_var_shots = accs_prob_var_shots.mean(axis=1)

    plt.figure()
    plt.loglog(shots, avg_accs_pl[0], label="langevin", c="orange")
    plt.loglog(shots, avg_accs_prob_var_shots, label="prob,$\lambda$=n_shots/2", c="red")

    main_color_prob = "b"
    colors_prob = [lighten_color(main_color_prob, amount=s) for s in np.linspace(1.6, 0.2, len(lambdas))]
    for i in range(len(lambdas)):
        plt.loglog(shots, avg_accs_prob[i], label=rf"prob,$\lambda$={lambdas[i]}", c=colors_prob[i])
    plt.legend()
    plt.xlabel("Number of shots [#]")
    plt.ylabel("$L_2$ squared error")
    plt.title("Comparison of accuracy wrt shots, different lambda values, loglog")
    if savefig:
        plt.savefig(f"shots_acc_comp_shots{ext}_lambda_prob_with_var_lambda_loglog.pdf", bbox_inches="tight")
    plt.show()

    

if __name__ =="__main__":
    main()