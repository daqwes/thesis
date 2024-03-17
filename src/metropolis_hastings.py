from typing import Iterable, Optional, Tuple
import numpy as np
from numpy.linalg import eig
import time 
import sys

parent_module = sys.modules['.'.join(__name__.split('.')[:-1]) or '__main__']
if __name__ == '__main__' or parent_module.__name__ == '__main__':
    from data_generation import generate_data, random_complex_ortho
    from data_generation_exact import init_matrices, get_true_rho, compute_measurements, compute_rho_inversion, random_uniform, random_multivariate_complex, get_measurables, random_standard_exponential, projectors_py, generate_data_exact
    from proj_langevin import gen_init_point
    from utils import compute_error
else:
    from .data_generation import generate_data, random_complex_ortho
    from .data_generation_exact import init_matrices, get_true_rho, compute_measurements, compute_rho_inversion, random_uniform, random_multivariate_complex, get_measurables, random_standard_exponential, projectors_py, generate_data_exact
    from .proj_langevin import gen_init_point
import functools
import itertools

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

# def compute_rho_inversion(n: int, p_as: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
#     """Compute the rho estimate with the inversion method
#     Args:
#         p_as (np.ndarray[R*A]): Vector mapping each observable and result combination to its empirical probability
#         P_rab: (np.ndarray[I=2^n x 3^n, J=4^n])
#         sig_b: (np.ndarray[J, d=16, d])
#     Returns:
#         Tuple[np.ndarray[d=2^n, d], np.ndarray]: the approximation of rho using the inversion technique, and its eigenvectors 
#     """
#     J = 4**n # Matches the number of base
#     I = 6**n # Matches  R*A = 2^n * 3^n = 6^n
#     d = R = 2**n # matrix dimension and number of possibilities for R^a_s ({-1, 1}^n)
#     A = 3**n # Number of possible measurements
#     npa = np.array
#     sx = np.array([[0, 1], [1, 0]])
#     sy = np.array([[0, 1j], [-1j, 0]])
#     sz = np.array([[1, 0], [0, -1]])
#     basis = np.stack([np.eye(2), sx, sy, sz])
#     b = npa(list(itertools.product(range(4), repeat=n))) # {I, x, y, z}^n
#     a = npa(list(itertools.product(range(1, 4), repeat=n))) # {x,y,z}^n
#     r = npa(list(itertools.product([-1, 1], repeat=n)))
#     sig_b = npa([functools.reduce(np.kron, (basis[b[i,:], :, :])) for i in range(J)])
#     P_rab = np.zeros((I, J))
#     for j in range(J):
#         tmp = np.zeros((R, A))
#         for s in range(R):
#             for l in range(A):
#                 val = np.prod(r[s, b[j] != 0])\
#                     * np.prod(a[l, b[j] != 0] == b[j, b[j]!=0])
#                 tmp[s,l] = val
#         P_rab[:, j] = tmp.flatten(order="F")
#     temp1 = p_as @ P_rab
#     temp1 = temp1/d

#     # Calculate coefs rho_b
#     rho_b = [0] * J
#     for i in range(J):
#         rho_b[i] = temp1[i]/(3**((b[i] == 0).sum()))

#     # Calculate density using inversion technique
#     rho_hat = np.zeros((d, d), dtype=np.complex128)
#     for s in range(J):
#         rho_hat += rho_b[s] * sig_b[s]

#     u_hat = eig(rho_hat)[1]

#     # renormalize lambda_hat
#     lamb_til = eig(rho_hat)[0]
#     lamb_til[lamb_til < 0] = 0
#     lamb_hat = lamb_til/lamb_til.sum()
#     return rho_hat, u_hat


def MH_prob(n: int, p_as: np.ndarray, Pra_m: np.ndarray, u_hat: np.ndarray, gamm: float, pkl: str|None, seed: int, n_iter: int = 500, n_burnin: int = 100) -> np.ndarray:
    """
    """
    if seed is not None:
        np.random.seed(seed)
    d = 2**n
    rho = np.zeros((d, d), dtype=np.complex128)
    Te = random_standard_exponential(d, seed) # Initial Y_i^0
    U = u_hat # eigenvectors of \hat(rho) found using inversion, initial V_i^0
    Lamb = Te/Te.sum() # gamma^0
    ro = 1/2
    be = 1
    cum_times = [0] * (n_iter) 
    rhos_record = np.zeros((n_iter, d, d), dtype=np.complex128)
    start_time = time.perf_counter()
    for t in range(n_iter):
        for j in range(d): # Loop for Y_i
            Te_can = Te.copy()
            Te_can[j] = Te[j] * np.exp(be * random_uniform(-0.5, 0.5, 1, seed)) # \tilde(Y)_i = exp(y ~ U(-0.5, 0.5)) Y_i^t-1
            L_can = Te_can/Te_can.sum() # \tilde(gamma)_i = \tilde(Y_i)/sum_j^d(\tilde(Y_j))
            tem_can = (U @ np.diag(L_can) @ np.conj(U.T)).flatten(order="F") # gamma * U * U^T (U = V in paper)
            tem = (U @ np.diag(Lamb) @ np.conj(U.T)).flatten(order="F") # prev gamma * U * U^T
            # ------------------
            ss1 = (Pra_m @ tem_can - p_as)**2  # l^prob: sum_a sum_s (Tr(v P^a_s) - hat(p^_a,s))^2
            ss2 = (Pra_m @ tem - p_as)**2
            # ------------------
            ss = (ss1 - ss2).sum()
            r_prior = (ro-1) * np.log(Te_can[j]/Te[j]) - Te_can[j] + Te[j] # other part of R acceptance ratio
            ap = -gamm*np.real(ss) # other part (why use np.real?)
            if np.log(random_uniform(0, 1, 1, seed=seed)) <= ap + r_prior: Te = Te_can # if value above draw from U(0,1), then update
            Lamb = Te/Te.sum() # gamma
        for j in range(d): # Loop for V_i
            U_can = U.copy()
            rd_U = U[:,j] + random_multivariate_complex(np.zeros(d), np.eye(d), 1, seed)/100#np.random.multivariate_normal(np.zeros(d*2),np.eye(d*2)/100, size=(1)).view(np.complex128)
            U_can[:, j] = norm_complex(rd_U) # Sample U/V from the unit sphere (not sure why we add to previous value)
            tem_can = (U_can @ np.diag(Lamb) @ np.conj(U_can.T)).flatten(order="F") # gamma * U * U^T
            tem = (U @ np.diag(Lamb) @ np.conj(U.T)).flatten(order="F") # gamma * U_t-1 * U^T_t-1
            ss1 = (Pra_m @ tem_can - p_as)**2
            ss2 = (Pra_m @ tem - p_as)**2
            ss = (ss1 - ss2).sum()
            ap = -gamm * np.real(ss) # other part of A accep ratio
            if np.log(random_uniform(0, 1, 1, seed)) <= ap: U = U_can # if value above draw from U(0,1), then update

        if t >= n_burnin:
            rho = U @ np.diag(Lamb) @ np.conj(U.T)/(t - n_burnin + 1) + rho*(1-1/(t-n_burnin + 1)) # approximate rho each time as rho_t = gamma_t * V_t * V_t^T /(t-n_burnin) + rho_t-1 / (1 - 1/(t-n_burnin)) -> the later we are, the more importance we give to prev rho
            rhos_record[t, :, :] = rho.copy()
        else:
            rhos_record[t, :, :] = U @ np.diag(Lamb) @ np.conj(U.T)
        cum_times[t] = time.perf_counter() - start_time
    return rhos_record, rho, cum_times

def run_MH(n: int, n_exp: int, n_shots: int, rho_true: np.ndarray, As: np.ndarray, y_hat: np.ndarray, n_iter: int = 500, n_burnin: int = 100, seed: int = 0):
    """Runner function for the prob-estimator
    Args:
        n (int): number of qubits
        n_exp (int): number of experiments
        n_shots (int): number of measurements
        rho_true (np.ndarray): true denstiy matrix, if available
        As (np.ndarray): measurement matrices
        y_hat (np.ndarray): empirical probabilities associated to each measurement matrix
        n_iter (int): number of iterations
        n_burnin (int): number of iterations for burnin
    Returns:
        np.ndarray: approximated version of the density matrix
    """
    if seed is not None:
        np.random.seed(seed)
    gamm = n_shots/2 # lambda in paper
    # TODO There should be a better way (more fair) to create the initial candidate
    # u_hat = random_complex_ortho()

    # TODO: this is done in order to have the same initial point in both algos, change later
    d = 2**n
    u_hat = gen_init_point(d, d) 
    rhos_record, rho_prob, cum_times = MH_prob(n, y_hat, As, u_hat, gamm, None, seed = None, n_iter=n_iter, n_burnin=n_burnin)
    return rhos_record, rho_prob, cum_times

def run_MH_exact(n: int, n_exp: int, n_shots: int, rho_true: np.ndarray, As: np.ndarray, y_hat: np.ndarray, n_iter: int = 500, n_burnin: int = 100, seed: int = 0):
    """Runner function for the prob-estimator
    Args:
        n (int): number of qubits
        n_exp (int): number of experiments
        n_shots (int): number of measurements
        rho_true (np.ndarray): true denstiy matrix, if available
        As (np.ndarray): measurement matrices
        y_hat (np.ndarray): empirical probabilities associated to each measurement matrix
        n_iter (int): number of iterations
        n_burnin (int): number of iterations for burnin
    Returns:
        np.ndarray: approximated version of the density matrix
    """
    d = 2**n
    if seed is not None:
        np.random.seed(seed)
    
    if n_shots is None:
        gamm = 1e7/2
    else:
        gamm = n_shots/2
    u_hat = gen_init_point(d, d, seed=None)
    # rho_hat, u_hat = compute_rho_inversion(n, b, y_hat, P_rab, sig_b)
    rhos_record, rho_prob, cum_times = MH_prob(n, y_hat, As, u_hat, gamm, pkl=None, seed=None, n_iter=n_iter, n_burnin=n_burnin)
    return rhos_record, rho_prob, cum_times

def main():
    seed = 0
    n = 3
    d = 2**n
    n_exp = d * d
    n_shots = 100000
    rho_type = "rank2"
    rho_true, As, y_hat = generate_data(n, n_exp, n_shots, rho_type=rho_type)
    # As_m1 = (As.T).reshape(4**n, -1)
    As_flat = np.zeros((4**n, 2**n * 2**n), dtype = np.complex128)
    for i in range(4**n):
        As_flat[i,:] = As[:,:,i].flatten(order="C") 
        # TODO: it is not clear why this works better than `flatten(order="F")`
        # as it is more correct to use the latter (similar to what is done in R)
    
    # rho_hat, u_hat = compute_rho_inversion(n, y_hat)

    gamm = n_shots/2 # lambda in paper
    n_iter = 2000
    n_burnin = 100
    u_hat = random_complex_ortho(d, d)
    rho_prob = MH_prob(n, y_hat, As_flat, u_hat, gamm, None, seed, n_iter, n_burnin)
    err = np.linalg.norm(rho_prob- rho_true, "fro")
    print(err**2)
    # print(rho_true[:4, :4])
    # print(rho_prob[:4, :4])
    # print(y_hat.shape)

def main_exact_data_gen():
    seed = 0
    n = 3
    d = 2**n
    n_exp = d*d
    n_iter = 600
    n_burnin = 100
    n_shots = 2000
    if n_shots is None:
        gamm = 1e7/2
    else:
        gamm = n_shots/2
    # Pra, sig_b, P_rab, b, a, r = init_matrices(n)
    rho_type = "rank2" 
    rho_true, As, y_hat = generate_data_exact(n, n_exp, n_shots, rho_type=rho_type, seed=seed)
    # Pra = get_measurables(n)
    # rho_true = get_true_rho(n, rho_type, seed=seed)
    # u_hat = random_complex_ortho(d, d, seed=seed)
    # # np.random.seed()
    # y_hat = compute_measurements(n, rho_true, n_shots, seed=None)
    prob_seed = None
    np.random.seed()
    u_hat = random_complex_ortho(d, d)
    # rho_hat, u_hat = compute_rho_inversion(n, b, y_hat, P_rab, sig_b)
    rhos_record, rho_prob, cum_times = MH_prob(n, y_hat, As, u_hat, gamm, None, seed=prob_seed, n_iter=n_iter, n_burnin=n_burnin)

    err_mse = compute_error(rho_prob, rho_true, "MSE")
    err_fro_sq = compute_error(rho_prob, rho_true, "fro_sq")
    print(f"MSE: {err_mse}")
    print(f"Fro^2: {err_fro_sq}")

    
if __name__ == "__main__":
    main_exact_data_gen()