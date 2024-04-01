import numpy as np
from scipy.linalg import sqrtm
from scipy.linalg import solve
import scipy
import time

import sys
parent_module = sys.modules['.'.join(__name__.split('.')[:-1]) or '__main__']
if __name__ == '__main__' or parent_module.__name__ == '__main__':
    from data_generation import generate_data, random_unitary
else:
    from .data_generation import generate_data, random_unitary

def complex_to_real(X: np.ndarray) -> np.ndarray:
    """ """
    X_re = np.real(X)
    X_im = np.imag(X)
    return np.sqrt(2) / 2 * np.block([[X_re, X_im], [-X_im, X_re]])

def real_to_complex(X: np.ndarray):
    """"""
    two_d, _ = X.shape
    d = int(two_d/2)
    return np.sqrt(2) * (X[:d,:d] + 1j*X[:d, d:])

def gen_init_point(d, r, seed=None) -> np.ndarray:
    """
    """
    # Generate initial candidate, VV* = V*V = I (columns are orthonormal wrt to the conj transpose)
    V = random_unitary(d, r, seed)
    # Generate candidate from the dirichlet distribution with all parameters equal to a same small constant (1/r)
    # This is equivalent to sampling from a Gamma, and then normalizing
    gamma0 = np.random.gamma(1 / r, 1, r)
    D = np.diag(gamma0) / gamma0.sum()
    # Multiply our unitary matrix of rank r by sqrt(D) (=singular values) to get a rank 1 (not true in practice, but at least in theory)
    Y_rho = V @ np.sqrt(D)
    return Y_rho

def f(
    Y_rho_r: np.ndarray,
    As_r: np.ndarray,
    y_hat: np.ndarray,
    lambda_: float,
    theta: float,
    alpha: float,
    As_r_swap: np.ndarray
):
    n_exp = y_hat.shape[0]
    s1, s2 = Y_rho_r.shape
    d, r = int(s1 / 2), int(s2 / 2)
    Y_rho_r_outer = Y_rho_r @ np.conj(Y_rho_r.T)
    y = np.trace(As_r_swap @ Y_rho_r_outer, axis1=1, axis2=2)
    # y = np.zeros(n_exp)
    # for j in range(n_exp):
    #     y[j] = np.trace(As_r[:, :, j] @ (Y_rho_r @ np.conj(Y_rho_r.T)))
    # print(np.allclose(y, y_vec))
    return lambda_ * np.linalg.norm(y_hat - np.sqrt(2) * y) ** 2 + alpha * (
        (2 * d + r + 2)
        * np.log(
            np.linalg.det(theta**2 * np.eye(2 * d) / np.sqrt(2)
                + np.sqrt(2) * Y_rho_r_outer))
                / 4 
                + (2 * d + r + 2) * d * np.log(2) / 4
            )
def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)

def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)

def gradf(
    Y_rho_r: np.ndarray,
    As_r: np.ndarray,
    y_hat: np.ndarray,
    lambda_: float,
    theta: float,
    alpha: float,
    As_r_swap: np.ndarray,
    As_r_sum_swap: np.ndarray
):
    s1, s2 = Y_rho_r.shape
    d, r = int(s1 / 2), int(s2 / 2)
    G = np.zeros_like(Y_rho_r)
    # print(Y_rho_r.dtype)
    
    Y_rho_r_outer = Y_rho_r @ np.conj(Y_rho_r.T)

    p1 = 2 * np.sqrt(2) * lambda_ * \
        (y_hat - np.sqrt(2) * np.trace(As_r_swap @ Y_rho_r_outer, axis1=1, axis2=2))
    p2 = As_r_sum_swap @ Y_rho_r
    # print(p1.shape, p2.shape)
    G = -np.sum(np.expand_dims(p1, (1, 2)) * p2, 0)
    # for j in range(n_exp):
    #     G -= (2 * np.sqrt(2) * lambda_ * 
    #             (y_hat[j] - np.sqrt(2) * np.trace(As_r[:, :, j] @ Y_rho_r_outer))
    #             * (As_r[:, :, j] + np.conj(As_r[:,:,j].T)) @ Y_rho_r)
    # print(np.allclose(G_vec, G))

    # the matrix is real, symmetric and positive definite
    to_inv = np.eye(2 * r) + 2 * (np.conj(Y_rho_r.T) @ Y_rho_r) / (theta**2)
    D, V = scipy.linalg.eigh(to_inv)
    A = (V * np.sqrt(D)) @ V.T
    
    # A = sqrtm(to_inv)
    # A = np.real(A)
    # print(A.dtype)

    b = (Y_rho_r.T)
    M = solve(A, b, assume_a="pos")
    M = np.conj(M.T) @ M
    M[:d, :d] = (M[:d, :d] + M[d : 2 * d, d : 2 * d]) / 2
    G += alpha * (
        (2 * d + r + 2) / theta**2 * (np.eye(2 * d) - 2 * M / theta**2) @ Y_rho_r
    )

    G[:d, :r] = (G[:d, :r] + G[d : 2 * d, r : 2 * r]) / 2
    G[d : 2 * d, r : 2 * r] = G[:d, :r]

    G[:d, r : 2 * r] = (G[:d, r : 2 * r] - G[d : 2 * d, :r]) / 2
    G[d : 2 * d, :r] = -G[:d, r : 2 * r]
    return G


def langevin(
    Y_rho0: np.ndarray,
    y_hat: np.ndarray,
    As: np.ndarray,
    r: int,
    n: int,
    n_exp: int,
    n_iter: int,
    n_burnin: int,
    alpha: float,
    lambda_: float,
    theta: float,
    beta: float,
    eta: float,
    seed: int,
):
    """
    Runs the langevin algorithm. 
    Expects the observables tensor in the [d, d, n_exp] format.
    """
    if seed is not None:
        np.random.seed(seed)
    d = 2**n

    Y_rho = Y_rho0
    # Apply change of variable
    Y_rho_r = complex_to_real(Y_rho)

    n_obs = As.shape[-1]
    As_r = np.zeros((2 * d, 2 * d, n_obs))
    for j in range(n_obs):
        As_r[:, :, j] = complex_to_real(As[:, :, j])

    # As_r is real now, no need to use complex dtypes
    As_r_swap = np.empty(As_r.shape[::-1], dtype=np.float64)
    As_r_sum_swap = np.empty((As_r.shape[::-1]),dtype=np.float64)
    for j in range(n_obs):
        As_r_swap[j,:,:] = As_r[:,:,j]
        As_r_sum_swap[j,:,:] = As_r[:, :, j] + np.conj(As_r[:,:,j].T)


    # Start training
    cost = np.zeros(n_iter + 1)
    n_rec = np.zeros(n_iter + 1)
    # Change Y_rho_r_record to Y_rho_record as they are complex
    Y_rho_record = np.zeros((n_iter, d, d), dtype=np.complex128)
    t_rec = np.zeros(n_iter)
    cost[0] = f(Y_rho_r, As_r, y_hat, lambda_, theta, alpha, As_r_swap)
    k = 0
    Y_rho_record[0, :, :] = Y_rho0
    t_start = time.perf_counter()
    for k in range(1, n_iter + 1):
        G = gradf(Y_rho_r, As_r, y_hat, lambda_, theta, alpha, As_r_swap, As_r_sum_swap)
        n_rec[k - 1] = np.linalg.norm(G, "fro")

        if (k - 1) % 1000 == 0:
            print(f"Iteration {k}, f = {cost[k-1]:4.2e} norm grad = {n_rec[k-1]:4.2e}")

        N1 = np.random.standard_normal((d, r))
        N2 = np.random.standard_normal((d, r))
        N = np.block([[N1, N2], [-N2, N1]])
        Y_rho_r -= eta * G + np.sqrt(2 * eta / beta) * N
        cost[k] = f(Y_rho_r, As_r, y_hat, lambda_, theta, alpha, As_r_swap)
        t_rec[k - 1] = time.perf_counter() - t_start            
        # Convert back to the real domain
        M = np.sqrt(2) * (Y_rho_r @ np.conj(Y_rho_r.T))
        Y_rho_record[k - 1, :, :] = real_to_complex(M)

    return Y_rho_record, t_rec, n_rec


def run_PL(n: int, n_exp: int, n_shots: int, rho_type: str, As: np.ndarray, y_hat: np.ndarray, n_iter: int = 5000, n_burnin: int = 100, seed = 0, running_avg: bool = False):
    """Runner function for the prob-estimator
    Args:
        n (int): number of qubits
        n_exp (int): number of experiments, corresponds to the number of measurement matrices 
                    (usually d*d for the full case)
        n_shots (int): number of measurements
        rho_type (str): type of true density matrix
        As (np.ndarray): measurement matrices
        y_hat (np.ndarray): empirical probabilities associated to each measurement matrix
        n_iter (int): number of iterations
        n_burnin (int): number of iterations to keep at the end
    Returns:
        np.ndarray: approximated version of the density matrix
    """
    d = 2**n
    r = d # TODO: change, not the most optimal way for the rank of the matrix
    if seed is not None:
        np.random.seed(seed)
    Y_rho0 = gen_init_point(d, r, seed = None)

    lambda_ = n_shots / 2
    eta = 0.05 /  n_shots
    alpha = 1

    if n == 3:
        theta = 0.1
        beta = 1e2
        # r = 6
    elif n == 4:
        theta = 1
        beta = 1e3
        # r = 2
    else:
        theta = 1
        beta = 1e3
        # r = 5

    # Y_rho_record, t_rec do not contain the samples/values for the burnin phase 
    Y_rho_record, t_rec, norm_rec = langevin(
        Y_rho0, y_hat, As, r, n, n_exp, n_iter, n_burnin, alpha, lambda_, theta, beta, eta, seed=None
    )
    if running_avg:
        M_avg = np.zeros_like(Y_rho_record[0,:,:])
        for i in range(n_iter - n_burnin):
            M_avg = Y_rho_record[n_burnin + i,:,:] * 1/(i + 1) + M_avg  * (1 - 1 /(i+1))
            # print((M_avg - Y_rho_record[n_burnin + i]).abs().sum())
            Y_rho_record[n_burnin + i] = M_avg.copy()
    else:
        M_avg = np.mean(Y_rho_record[n_burnin:,:,:], 0) # 
    return Y_rho_record, M_avg, t_rec

def main():
    n = 3
    d = 2**n
    n_exp = d * d
    n_shots = 2000
    n_iter = 2000
    n_burnin = 1000
    seed = 0
    rho_type = "rank2"
    rho_true, As, y_hat = generate_data(n, n_exp, n_shots, rho_type, seed= seed)

    rhos_pl, rho_avg_pl, cum_times_pl  = run_PL(n, n_exp, n_shots, rho_type, As, y_hat, n_iter, n_burnin, seed=seed, running_avg=True)
    err = np.linalg.norm(rho_avg_pl - rho_true, "fro")
    print(err**2)



if __name__ == "__main__":
    main()
