import time
import numpy as np
from scipy.special import binom

from data_generation_exact import random_multivariate_complex, random_standard_exponential, random_uniform, generate_data_exact_PL
from data_generation import norm_complex
from proj_langevin import gen_init_point, complex_to_real, real_to_complex

def eval_posterior_real(Y_r: np.ndarray, As_r_swap: np.ndarray, y_hat: np.ndarray, lambda_: float, theta: float, log_transform: bool):
    s1, s2 = Y_r.shape
    d, r = int(s1 / 2), int(s2 / 2)
    # print(np.linalg.norm(Y_r, 'fro'))
    Y_rho_r_outer = Y_r @ np.conj(Y_r.T)
    y = np.trace(As_r_swap @ Y_rho_r_outer, axis1=1, axis2=2)
    
    if log_transform:
        lik = lambda_ * np.linalg.norm(y_hat - np.sqrt(2) * y) ** 2
        prior = (2*d + r + 2)/4 * np.log(np.linalg.det(theta**2 * np.eye(2 * d) / np.sqrt(2)
                    + np.sqrt(2) * Y_rho_r_outer))
        post = -(lik + prior)
    # post = exp(-f) with f = L + log(P) -> exp(-L)/P
    else:
        lik = np.exp(
            - lambda_ * np.linalg.norm(y_hat - np.sqrt(2) * y) ** 2
        )
        prior = np.linalg.det(theta**2 * np.eye(2 * d) / np.sqrt(2)
                    + np.sqrt(2) * Y_rho_r_outer) ** ((2*d + r + 2)/4)
        print(np.linalg.norm(y_hat - np.sqrt(2) * y) ** 2, prior)
        post = lik/prior
    return post

def eval_posterior_complex(Y: np.ndarray, As: np.ndarray, y_hat: np.ndarray, lambda_: float, theta: float):
    d, r = Y.shape
    lik = np.exp(
        lambda_ * ...
    )
    prior = ...
    return lik * prior
    
def eval_proposal(Y_new: np.ndarray, Y_old: np.ndarray, dist: str = "normal"):
    # TODO
    m, n = Y_new.shape
    if dist == "normal":
        return 1/(2 * np.pi)**(m*n) * np.exp(-1/2 * np.linalg.norm(Y_new, ord="fro")**2)
    elif dist == "normal_dep":
        return 1/(2 * np.pi)**(m*n) * np.exp(-1/2 * np.linalg.norm(Y_new - Y_old, ord="fro")**2)
    elif dist == "goe":
        assert m == n, "d and r must have the same value for GOE"
        Z = (4 * np.pi)**(m/2) * (2*np.pi)**(1/2 * binom(m, 2))
        return np.exp(-1/4 * np.trace(Y_new @ Y_new)) / Z
def sample_proposal(d: int, r: int, Y_curr: np.ndarray, seed: int, dist: str = "normal"):
    """
    TODO
    In order for the sample to be valid, it needs to respect the physical constraints of the system.
    - Symmetric/Hermitian
    - Semi-definite positive
    - Trace = 1
    This means that it needs to come from the complex hypersphere, which amounts to it being of frob norm = 1
    """
    if seed is not None:
        np.random.seed(seed)
    if dist == "normal":
        Y_prop = np.random.randn(d, r)
        return Y_prop
    elif dist == "normal_dep":
        Y_prop = np.random.randn(d, r)
        return Y_prop + Y_curr
    elif dist == "goe":
        assert d == r, "d and r must have the same value for GOE"
        G = np.random.randn(d, r)
        return (G+G.T)/(np.sqrt(2))
    elif dist == "unitary_rank1":
        Y_prop = gen_init_point(d, r, seed)
        return Y_prop/Y_prop.sum()

def acc_rate(Y_next: np.ndarray, Y_prev: np.ndarray, As: np.ndarray, As_r_swap: np.ndarray, y_hat: np.ndarray, lambda_: float, theta: float, prop_dist: str = "normal", log_transform: bool = False) -> float:
    if log_transform:
        prop = np.log(eval_proposal(Y_prev, prop_dist)) - np.log(eval_proposal(Y_next, prop_dist))
        post = eval_posterior_real(Y_next, As_r_swap, y_hat, lambda_, theta, log_transform=True)\
               - eval_posterior_real(Y_prev, As_r_swap, y_hat, lambda_, theta, log_transform=True)
        ratio = prop + post
    else:
        n = eval_proposal(Y_prev, prop_dist) * eval_posterior_real(Y_next, As_r_swap, y_hat, lambda_, theta)
        d = eval_proposal(Y_next, prop_dist) * eval_posterior_real(Y_prev, As_r_swap, y_hat, lambda_, theta) 
        # print(eval_proposal(next, prop_dist), eval_posterior_real(prev, As_r_swap, y_hat, lambda_, theta ))
        # print(eval_proposal(prev, prop_dist), eval_posterior_real(next, As_r_swap, y_hat, lambda_, theta))
        ratio = n/d
    return min(1, ratio)


def MH_studentt(n: int, y_hat: np.ndarray, As: np.ndarray, Y0: np.ndarray, lambda_: float, theta: float, seed: int, n_iter: int = 500, n_burnin: int = 100, ) -> np.ndarray:
    """ TODO: try to implement with prior of the form:
        C_theta * det(theta^2*I_d + YY*)^-(2d+r+2)/2    
    """
    if seed is not None:
        np.random.seed(seed)
    d, r = Y0.shape
    # Apply change of variable
    Y0_r = complex_to_real(Y0)

    n_obs = As.shape[-1]
    As_r = np.zeros((2 * d, 2 * d, n_obs))
    for j in range(n_obs):
        As_r[:, :, j] = complex_to_real(As[:, :, j])

    # As_r is real now, no need to use complex dtypes
    As_r_swap = np.empty(As_r.shape[::-1], dtype=np.float64)
    for j in range(n_obs):
        As_r_swap[j,:,:] = As_r[:,:,j]

    # rho = np.zeros((d, d), dtype=np.complex128)
    rhos_record = np.zeros((n_iter, d, d), dtype=np.complex128)
    cum_times = np.zeros(n_iter)
    Y_r_prev = Y0_r
    rho0 = np.sqrt(2) * (Y_r_prev @ np.conj(Y_r_prev.T))
    rhos_record[0,:,:] = real_to_complex(rho0)

    start_time = time.perf_counter()
    proposal_dist = "normal_dep"
    for t in range(n_iter):
        # Shape should be (2d, 2r) as we now work in the real domain
        Y_r_next = sample_proposal(int(2*d), int(2*r), Y_r_prev, None, dist=proposal_dist)
        rd = np.random.random()
        if rd < acc_rate(Y_r_next, Y_r_prev, As, As_r_swap, y_hat, lambda_, theta, proposal_dist, log_transform=True):
            Y_r_prev = Y_r_next
        # if t >= n_burnin and run_avg:
        #     rho = prev/(t - n_burnin + 1) + rho*(1-1/(t-n_burnin + 1)) # approximate rho each time as rho_t = gamma_t * V_t * V_t^T /(t-n_burnin) + rho_t-1 / (1 - 1/(t-n_burnin)) -> the later we are, the more importance we give to prev rho
        #     rhos_record[t, :, :] = rho.copy()
        # else:
        rho_iter = np.sqrt(2) * (Y_r_prev @ np.conj(Y_r_prev.T))
        rhos_record[t, :, :] = real_to_complex(rho_iter)
        cum_times[t] = time.perf_counter() - start_time
        # exit(0)
    return rhos_record, rho_iter, cum_times


def run_MH_studentt(n: int, n_shots: int, As: np.ndarray, y_hat: np.ndarray, n_iter: int = 500, n_burnin: int = 100, seed: int = None, run_avg: bool = False):
    if seed is not None:
        np.random.seed(seed)
    d = 2**n
    r = d # TODO: change as not the most optimal approach
    Y0 = gen_init_point(d, r)

    lambda_ = n_shots / 2
    if n == 3:
        theta = 0.1
        # r = 6
    elif n == 4:
        theta = 1
        # r = 2
    else:
        theta = 1
    rhos_record, rho_mh_stt, cum_times = MH_studentt(n, y_hat, As, Y0, lambda_, theta, seed, n_iter, n_burnin)
    
    if run_avg:
        M_avg = np.zeros_like(rhos_record[0,:,:])
        for i in range(n_iter - n_burnin):
            M_avg = rhos_record[n_burnin + i,:,:] * 1/(i + 1) + M_avg  * (1 - 1 /(i+1))
            # print((M_avg - Y_rho_record[n_burnin + i]).abs().sum())
            rhos_record[n_burnin + i] = M_avg.copy()
    else:
        M_avg = np.mean(rhos_record[n_burnin:,:,:], 0) # 

    return rhos_record, M_avg, cum_times

def main():
    seed = 0
    n = 3
    n_exp = 3**n
    n_shots = 2000
    n_iter = 5000
    n_burnin = 1000
    run_avg = True
    rho_true, As, y_hat = generate_data_exact_PL(n, n_exp, n_shots, rho_type="rank2", seed= seed)
    np.random.seed()
    rhos_mh_stt, rho_mh_stt, cum_times_mh_stt  = run_MH_studentt(n, n_shots, As, y_hat, n_iter, n_burnin, seed = None, run_avg=run_avg)
    print(rho_true)
    print(rho_mh_stt)
    err = np.linalg.norm(rho_mh_stt - rho_true, "fro")
    print(err**2)

if __name__ == "__main__":
    main()