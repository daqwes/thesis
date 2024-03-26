import time
import numpy as np

from data_generation_exact import random_multivariate_complex, random_standard_exponential, random_uniform
from data_generation import norm_complex


def eval_posterior(Y: np.ndarray):
    # TODO
    pass
def eval_proposal(Y: np.ndarray):
    # TODO
    pass

def sample_proposal(x):
    """
    TODO
    In order for the sample to be valid, it needs to respect the physical constraints of the system.
    This means that it needs to come from the complex hypersphere 
    """
    pass

def acc_rate(prev: np.ndarray, next: np.ndarray) -> float:
    n = eval_proposal(next) * eval_posterior(prev)
    d = eval_proposal(prev) * eval_posterior(next) 
    return min(1, n/d)


def MH_studentt(n: int, p_as: np.ndarray, Pra_m: np.ndarray, u_hat: np.ndarray, gamm: float, pkl: str|None, seed: int, n_iter: int = 500, n_burnin: int = 100, run_avg: bool = True) -> np.ndarray:
    """ TODO: try to implement with prior of the form:
        C_theta * det(theta^2*I_d + YY*)^-(2d+r+2)/2    
    """
    if seed is not None:
        np.random.seed(seed)
    d = 2**n
    rho = np.zeros((d, d), dtype=np.complex128)
    rhos_record = np.zeros((n_iter, d, d), dtype=np.complex128)
    cum_times = np.zeros(n_iter)
    prev = u_hat
    rhos_record[0,:,:] = prev
    start_time = time.perf_counter()
    for t in range(n_iter):
        next = sample_proposal()
        r = np.random.random()
        if r < acc_rate(prev, next):
            prev = next
        if t >= n_burnin and run_avg:
            rho = prev/(t - n_burnin + 1) + rho*(1-1/(t-n_burnin + 1)) # approximate rho each time as rho_t = gamma_t * V_t * V_t^T /(t-n_burnin) + rho_t-1 / (1 - 1/(t-n_burnin)) -> the later we are, the more importance we give to prev rho
            rhos_record[t, :, :] = rho.copy()
        else:
            rhos_record[t, :, :] = prev
        cum_times[t] = time.perf_counter() - start_time
    return rhos_record, rho, cum_times


def run_MH_studentt():

    MH_studentt()


if __name__ == "__main__":
    run_MH_studentt()