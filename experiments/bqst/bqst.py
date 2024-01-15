import numpy as np
from numpy.linalg import eig
import itertools
import functools
import time
from typing import *
import matplotlib.pyplot as plt
import pickle
import os
import glob
import pandas as pd
npa = np.array

sx = np.array([[0, 1], [1, 0]])
sy = np.array([[0, 1j], [-1j, 0]])
sz = np.array([[1, 0], [0, -1]])
basis = np.stack([np.eye(2), sx, sy, sz])

n = 4 # Nb of qubits
J = 4**n # Matches the number of bases
I = 6**n # Matches  R*A = 2^n * 3^n = 6^n
d = R = 2**n # matrix dimension and number of possibilities for R^a_s ({-1, 1}^n)
A = 3**n # Number of possible measurements

b = npa(list(itertools.product(range(4), repeat=n))) # {I, x, y, z}^n
a = npa(list(itertools.product(range(1, 4), repeat=n))) # {x,y,z}^n
r = npa(list(itertools.product([-1, 1], repeat=n)))


def norm_complex(arr: np.ndarray):
    """Normalizes complex vector or matrix, in which case it normalizes it row by row
    Args:
        arr (np.ndarray)
    Returns:
        np.ndarray
    """
    if len(arr.shape) > 1:
        ret_out = np.zeros_like(arr)
        for i in range(arr.shape[0]):
            ret_out[i,:] = arr[i]/np.sqrt((np.abs(arr[i])**2).sum())  
        return ret_out
    else:
        return arr/np.sqrt((np.abs(arr)**2).sum())

def projectors(idx_list, r_):
    """
    Returns the P^{a_i}_{s_i} list of projection matrices
    Note1: the evs returned by numpy and the ones obtained don't match, 
    but as they are not unique, it is still correct.
    """
    r_idx = [1 if i==-1 else 0 for i in r_]
    evs = [eig(basis[i])[1] for i in idx_list]
    selected_evs = np.array([ev[:, r_idx[i]] for i, ev in enumerate(evs)])
    ret = npa([np.outer(np.conj(ev), ev) for ev in selected_evs])
    return ret

def init_matrices() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Initializes all the matrices for the simulation of the data, the inversion method and for the prob-estimator
    Returns:
        Tuple[np.ndarray[A*R, d*d], np.ndarray[J, d*d], np.ndarray[I, J]]: Pra, sig_b, P_rab
    """
    # Corresponds to P^a_s in paper (each row here is a matrix), size: 2^n x 3^n flattened
    Pra = []
    for j in range(A):
        for i in range(R):
            Pra.append(npa(functools.reduce(np.kron, projectors(a[j], r[i]))).flatten("F"))
    Pra = npa(Pra)

    # Pauli basis for n qubit 
    sig_b_path = f"pkled/sig_b_{n}.pkl"
    if os.path.exists(sig_b_path):
        with open(sig_b_path, 'rb') as file:
            sig_b = pickle.load(file) 
    else:
        sig_b = npa([functools.reduce(np.kron, (basis[b[i,:], :, :])) for i in range(J)])
        with open(sig_b_path, 'wb') as file:
            pickle.dump(sig_b, file)
    # Only used for the calculation of rho_hat, size: 6^n x 4^n
    # Matrix P_{(r,a),b}
    P_rab_path = f"pkled/P_rab_{n}.pkl"
    if os.path.exists(P_rab_path):
        with open(P_rab_path, 'rb') as file:
            P_rab = pickle.load(file) 
    else:
        P_rab = np.zeros((I, J))
        for j in range(J):
            tmp = np.zeros((R, A))
            for s in range(R):
                for l in range(A):
                    val = np.prod(r[s, b[j] != 0])\
                        * np.prod(a[l, b[j] != 0] == b[j, b[j]!=0])
                    tmp[s,l] = val
            P_rab[:, j] = tmp.flatten(order="F")
        with open(P_rab_path, 'wb') as file:
            pickle.dump(P_rab, file)

    return Pra, sig_b, P_rab

def get_true_rho(rho_type: str = "rank1") -> np.ndarray:
    """ Sample true rho 
    Args:
        rho_type (str): type of density matrix we want to sample. Possibilities are
            - rank1
            - rank2
            - approx-rank2
            - rankd
    Returns:
        np.ndarray[d, d]: true density matrix
    """
    if rho_type == "rank1":
        # Pure state - rank1
        dens_ma = np.zeros((d, d), dtype="complex")
        dens_ma[0, 0] = 1
    elif rho_type == "rank2":
        # Rank2
        v1 = np.zeros(d, dtype="complex")
        v1[0:int(d/2)] = 1
        v1 = norm_complex(v1)
        v2 = np.zeros(d, dtype="complex")
        v2[int(d/2):d] = 1j
        v2 = norm_complex(v2)
        dens_ma = 1/2 * np.outer(v1, np.conj(v1)) + 1/2 * np.outer(v2, np.conj(v2))
    elif rho_type == "approx-rank2":
        # Approx rank2
        v1 = np.zeros(d, dtype="complex")
        v1[0:int(d/2)] = 1
        v1 = norm_complex(v1)
        v2 = np.zeros(d, dtype="complex")
        v2[int(d/2):d] = 1j
        v2 = norm_complex(v2)
        dens_ma = 1/2 * np.outer(v1, np.conj(v1)) + 1/2 * np.outer(v2, np.conj(v2))
        w = 0.98
        dens_ma = w * dens_ma + (1 - w) * np.eye(d)/d
    elif rho_type == "rankd":
        # Maximal mixed state (rankd = 16)
        u = norm_complex(np.random.multivariate_normal(np.zeros(d*2),np.eye(d*2)/100, size=(d)).view(np.complex128))
        dens_ma = np.conj(u.T) @ u /d
    else:
        raise ValueError("rho_type must be one of the possible types")
    return dens_ma

def read_true_data():
    """Reads and returns the empirical frequencies from a real world dataset
    Args:
        None
    Returns:
        np.ndarray[3^n x 2^n]
    """
    data_path = "data/W4-Data.dat"
    p_as = pd.read_table(data_path, header=None).to_numpy()[:, 1:].flatten()
    return p_as

def compute_measurements(dens_ma, n_meas: int=2000):
    """Simulate the data measurement process for the prob estimator
    Args:
        dens_ma (np.ndarray[d= 2^n, d]): True rho
        n_meas (int): Number of measurements
    Returns:
        np.ndarray[R*A]: Vector mapping each observable and result combination to its empirical probability
    """
    # Corresponds to Tr(rho \dot P^a_s), which in turn corresponds to p_a,s
    # These probabilities are the true ones, so we do not have access to them
    # We use them below to measure the system
    Prob_ar = np.zeros((A, R))
    if n==1:
        for i in range(A):
            for j in range(R):
                Prob_ar[i,j] = dens_ma.flatten(order="F") @ projectors(a[i], r[j])
    else:
        for i in range(A):
            for j in range(R):
                Prob_ar[i,j] = np.diag(dens_ma @ npa(functools.reduce(np.kron, projectors(a[i], r[j])))).sum()
    Prob_ar = np.real(Prob_ar)

    # For each observable, we sample n_meas samples 
    # according to the true probabilities calculated above, and then give the probability
    # For example: 
    # if n=4 (qubits) and a_i = {x, x, y, z}, then an outcome could be s_j {-1, 1, -1, 1}
    # For this pair of a,s, we calculate the number of times we sampled this situation (the H == s part) 
    # and get the empirical probability for this combination

    p_ra = np.zeros((R, A)) # = \hat{p}_a,s
    for i, x in enumerate(Prob_ar):
        H = np.random.choice(R, n_meas, replace=True, p=x) #n_meas elements
        out = []
        for s in range(R):
            out.append((H==s).sum()/n_meas) # Calculate the empirical prob of this combination
        p_ra[:, i] = out
    # Transform matrix to vector form
    return p_ra.flatten(order="F")

def compute_rho_inversion(p_as: np.ndarray, P_rab: np.ndarray, sig_b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the rho estimate with the inversion method
    Args:
        p_as (np.ndarray[R*A]): Vector mapping each observable and result combination to its empirical probability
        P_rab: (np.ndarray[I=2^n x 3^n, J=4^n])
        sig_b: (np.ndarray[J, d=16, d])
    Returns:
        Tuple[np.ndarray[d=2^n, d], np.ndarray]: the approximation of rho using the inversion technique, and its eigenvectors 
    """
    temp1 = p_as @ P_rab
    temp1 = temp1/d

    # Calculate coefs rho_b
    rho_b = [0] * J
    for i in range(J):
        rho_b[i] = temp1[i]/(3**((b[i] == 0).sum()))

    # Calculate density using inversion technique
    rho_hat = np.zeros((d, d), dtype=np.complex128)
    for s in range(J):
        rho_hat += rho_b[s] * sig_b[s]

    u_hat = eig(rho_hat)[1]

    # renormalize lambda_hat
    lamb_til = eig(rho_hat)[0]
    lamb_til[lamb_til < 0] = 0
    lamb_hat = lamb_til/lamb_til.sum()
    return rho_hat, u_hat

def MH_prob(p_as: np.ndarray, Pra: np.ndarray, u_hat: np.ndarray, n_meas: int, rho_type: str, ignore_pkl: bool, real: bool = False, n_iter: int = 500, n_burnin: int = 100) -> np.ndarray:
    """Estimate rho using MH and the prob-estimator likelihood
    Args:
        p_as (np.ndarray[R=2^n * A=3^n]):
        Pra (np.ndarray[R*A, 16x16=256]):
        u_hat (np.ndarray[d, d]):
        n_meas (int): number of measurements
        rho_type (str):
    Returns:
        np.ndarray[d=2^n, d]
    """
    path = f"pkled/prob_rho_{n}_{rho_type}.pkl"
    if os.path.exists(path) and not ignore_pkl and not real:
        with open(path, "rb") as file:
            rho = pickle.load(file)
        return rho
    rho = np.zeros((d, d))
    Te = np.random.standard_exponential(d) # Initial Y_i^0
    U = u_hat # eigenvectors of \hat(rho) found using inversion, initial V_i^0
    Lamb = Te/Te.sum() # gamma^0
    ro = 1/2
    be = 1

    gamm = n_meas/2 # lambda in paper 
    start_time = time.time()
    Pra_m = Pra.reshape((I, J))
    for t in range(n_iter + n_burnin):
        for j in range(d): # Loop for Y_i       
            Te_can = Te.copy() 
            Te_can[j] = Te[j] * np.exp(be * np.random.uniform(-0.5, 0.5, 1)) # \tilde(Y)_i = exp(y ~ U(-0.5, 0.5)) Y_i^t-1
            L_can = Te_can/Te_can.sum() # \tilde(gamma)_i = \tilde(Y_i)/sum_j^d(\tilde(Y_j))
            tem_can = (U @ np.diag(L_can) @ np.conj(U.T)).flatten(order="F") # gamma * U * U^T (U = V in paper)
            tem = (U @ np.diag(Lamb) @ np.conj(U.T)).flatten(order="F") # prev gamma * U * U^T
            ss1 = (Pra_m @ tem_can - p_as)**2 # l^prob: sum_a sum_s (Tr(v P^a_s) - hat(p^_a,s))^2
            ss2 = (Pra_m @ tem - p_as)**2
            ss = (ss1 - ss2).sum()
            r_prior = (ro-1) * np.log(Te_can[j]/Te[j]) - Te_can[j] + Te[j] # other part of R acceptance ratio
            ap = -gamm*np.real(ss) # other part (why use np.real?)
            if np.log(np.random.uniform(0, 1, 1)) <= ap + r_prior: Te = Te_can # if value above draw from U(0,1), then update
            Lamb = Te/Te.sum() # gamma
        for j in range(d): # Loop for V_i
            U_can = U.copy()
            U_can[:, j] = norm_complex(U[:,j] + np.random.multivariate_normal(np.zeros(d*2),np.eye(d*2)/100, size=(1)).view(np.complex128)) # Sample U/V from the unit sphere (not sure why we add to previous value)
            tem_can = (U_can @ np.diag(Lamb) @ np.conj(U_can.T)).flatten(order="F") # gamma * U * U^T
            tem = (U @ np.diag(Lamb) @ np.conj(U.T)).flatten(order="F") # gamma * U_t-1 * U^T_t-1
            ss1 = (Pra_m @ tem_can - p_as)**2
            ss2 = (Pra_m @ tem - p_as)**2
            ss = (ss1 - ss2).sum()
            ap = -gamm * np.real(ss) # other part of A accep ratio
            if np.log(np.random.uniform(0, 1, 1)) <= ap: U = U_can # if value above draw from U(0,1), then update

        if t > n_burnin:
            rho = U @ np.diag(Lamb) @ np.conj(U.T)/(t - n_burnin) + rho*(1-1/(t-n_burnin)) # approximate rho each time as rho_t = gamma_t * V_t * V_t^T /(t-n_burnin) + rho_t-1 / (1 - 1/(t-n_burnin)) -> the later we are, the more importance we give to prev rho
    end_time = time.time()
    print(f"Took: {end_time - start_time} s")
    if not real:
        with open(path, 'wb') as file:
            pickle.dump(rho, file)  
    return rho

def MH_dens(rho_hat: np.ndarray, u_hat: np.ndarray, ignore_pkl: bool, n_iter: int = 500, n_burnin: int = 100) -> np.ndarray:
    """Estimate rho using MH and the dens-estimator likelihood
    Args:
        p_as (np.ndarray[R=2^n * A=3^n]):
        Pra (np.ndarray[R*A, 16x16=256]):
        u_hat (np.ndarray[d, d]):
        n_meas (int): number of measurements
        rho_type (str):
    Returns:
        np.ndarray[d=2^n, d]
    """
    path = f"pkled/dens_rho_{n}.pkl"
    if os.path.exists(path) and not ignore_pkl:
        with open(path, "rb") as file:
            rho = pickle.load(file)
        return rho
    rho = np.zeros((d, d))
    Te = np.random.standard_exponential(d) # Initial Y_i^0
    U = u_hat # eigenvectors of \hat(rho) found using inversion, initial V_i^0
    Lamb = Te/Te.sum() # gamma^0
    ro = 1/2
    be = 1
    N = 2000
    gamm = N * A/4 # lambda in paper 
    start_time = time.time()
    for t in range(n_iter + n_burnin):
        for j in range(d): # Loop for Y_i       
            Te_can = Te.copy() 
            Te_can[j] = Te[j] * np.exp(be * np.random.uniform(-0.5, 0.5, 1)) # \tilde(Y)_i = exp(y ~ U(-0.5, 0.5)) Y_i^t-1
            L_can = Te_can/Te_can.sum() # \tilde(gamma)_i = \tilde(Y_i)/sum_j^d(\tilde(Y_j))
            tem_can = (U @ np.diag(L_can) @ np.conj(U.T)) # gamma * U * U^T (U = V in paper)
            tem = (U @ np.diag(Lamb) @ np.conj(U.T)) # prev gamma * U * U^T
            ss1 = (np.abs((tem_can - rho_hat))**2) # l^dens: sum_a sum_s (Tr(v P^a_s) - hat(p^_a,s))^2
            ss2 = (np.abs((tem - rho_hat))**2)
            ss = (ss1 - ss2).sum()
            r_prior = (ro-1) * np.log(Te_can[j]/Te[j]) - Te_can[j] + Te[j] # other part of R acceptance ratio
            ap = -gamm*np.real(ss) # other part (why use np.real?)
            if np.log(np.random.uniform(0, 1, 1)) <= ap + r_prior: Te = Te_can # if value above draw from U(0,1), then update
            Lamb = Te/Te.sum() # gamma
        for j in range(d): # Loop for V_i
            U_can = U.copy()
            U_can[:, j] = norm_complex(U[:,j] + np.random.multivariate_normal(np.zeros(d*2),np.eye(d*2)/100, size=(1)).view(np.complex128)) # Sample U/V from the unit sphere (not sure why we add to previous value)
            tem_can = (U_can @ np.diag(Lamb) @ np.conj(U_can.T)) # gamma * U * U^T
            tem = (U @ np.diag(Lamb) @ np.conj(U.T)) # gamma * U_t-1 * U^T_t-1
            ss1 = np.abs((tem_can - rho_hat))**2
            ss2 = np.abs((tem - rho_hat))**2
            ss = (ss1 - ss2).sum()
            ap = -gamm * np.real(ss) # other part of A accep ratio
            if np.log(np.random.uniform(0, 1, 1)) <= ap: U = U_can # if value above draw from U(0,1), then update

        if t > n_burnin:
            rho = U @ np.diag(Lamb) @ np.conj(U.T)/(t - n_burnin) + rho*(1-1/(t-n_burnin)) # approximate rho each time as rho_t = gamma_t * V_t * V_t^T /(t-n_burnin) + rho_t-1 / (1 - 1/(t-n_burnin)) -> the later we are, the more importance we give to prev rho
    end_time = time.time()
    print(f"Took: {end_time - start_time} s")
    with open(path, 'wb') as file:
        pickle.dump(rho, file)
    return rho


def plot_evs_sim(dens_ma, rho_mh, rho_inv, rho_type):
    evs_true = sorted(np.abs(eig(dens_ma)[0]), reverse=True)
    evs_MH = sorted(np.abs(eig(rho_mh)[0]), reverse=True)
    evs_inv = sorted(np.abs(eig(rho_inv)[0]), reverse=True)
    x = range(1, len(evs_true) + 1)
    plt.plot(x, evs_true, "-o", label="true eigenvalues")
    plt.plot(x, evs_MH, "--o", label="prob-estimator")
    plt.plot(x, evs_inv, "--o", label="inversion")
    plt.legend()
    plt.xticks(x)
    plt.title(rho_type)
    plt.xlabel("eigenvalues")
    plt.savefig(f"prob_{rho_type}.pdf", bbox_inches="tight")
    plt.show()

def plot_evs_real(rho_dens,rho_prob,rho_inv):
    evs_dens = sorted(np.abs(eig(rho_dens)[0]), reverse=True)
    evs_prob = sorted(np.abs(eig(rho_prob)[0]), reverse=True)
    evs_inv = sorted(np.abs(eig(rho_inv)[0]), reverse=True)
    x = range(1, len(evs_dens) + 1)
    plt.plot(x, evs_dens, "--o", label="dens-estimator")
    plt.plot(x, evs_prob, "--o", label="prob-estimator")
    plt.plot(x, evs_inv, "--o", label="inversion")
    plt.legend()
    plt.xticks(x)
    plt.title("True data (n=4)")
    plt.xlabel("eigenvalues")
    plt.savefig(f"real_dens_prob_4.pdf", bbox_inches="tight")
    plt.show()

def run_experiment_simulated(rho_type: str):
    Pra, sig_b, P_rab = init_matrices()

    ignore_pkl = True
    reset_pkl = False
    n_meas = 2000
    dens_ma = get_true_rho(rho_type)
    p_as = compute_measurements(dens_ma, n_meas)
    rho_hat, u_hat = compute_rho_inversion(p_as, P_rab, sig_b)
    n_iter = 500
    n_burnin = 100
    if not ignore_pkl:
        np.random.seed(0)
    if reset_pkl:
        for file in glob.glob("pkled/prob_rho_*.pkl"):
            os.remove(file)
    rho = MH_prob(p_as, Pra, u_hat, rho_type, ignore_pkl, n_iter, n_burnin)
    mean_rho = np.real(np.mean((dens_ma - rho) @ np.conj((dens_ma - rho).T)))
    mean_rho_hat = np.real(np.mean((dens_ma - rho_hat) @ np.conj((dens_ma - rho_hat).T)))
    print(f"MSE MH: {mean_rho:.2e} - MSE inversion: {mean_rho_hat:.2e}")
    plot_evs_sim(dens_ma, rho, rho_hat, rho_type)

def run_experiment_real():
    Pra, sig_b, P_rab = init_matrices()
    ignore_pkl = True
    reset_pkl = False
    p_as = read_true_data()
    rho_hat, u_hat = compute_rho_inversion(p_as, P_rab, sig_b)
    n_iter = 500
    n_burnin = 100
    if not ignore_pkl:
        np.random.seed(0)
    if reset_pkl:
        for file in glob.glob("pkled/dens_rho_*.pkl"):
            os.remove(file)
    rho_dens = MH_dens(rho_hat, u_hat, n_iter, n_burnin)
    rho_prob = MH_prob(p_as, Pra, u_hat, 2000, "none", ignore_pkl, True, n_iter, n_burnin)
    mean_rho_dens = np.real(np.mean((rho_dens - rho_hat) @ np.conj((rho_dens - rho_hat).T)))
    mean_rho_prob = np.real(np.mean((rho_prob - rho_hat) @ np.conj((rho_prob - rho_hat).T)))
    print(f"MSE MH dens: {mean_rho_dens:.2e}; MSE MH prob: {mean_rho_prob:.2e}")
    plot_evs_real(rho_dens, rho_prob, rho_hat)

def main():
    exp = "real"

    if exp=="sim":
        rho_types = ["rank1", "rank2", "approx-rank2", "rankd"]

        for rho_type in rho_types:
            run_experiment_simulated(rho_type)
    elif exp == "real":
        run_experiment_real()

if __name__ == "__main__":
    main()