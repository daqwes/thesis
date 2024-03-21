import time
import numpy as np

from data_generation_exact import random_multivariate_complex, random_standard_exponential, random_uniform
from data_generation import norm_complex

def MH_studentt(n: int, p_as: np.ndarray, Pra_m: np.ndarray, u_hat: np.ndarray, gamm: float, pkl: str|None, seed: int, n_iter: int = 500, n_burnin: int = 100) -> np.ndarray:
    """ TODO: try to implement with prior of the form:
        C_theta * det(theta^2*I_d + YY*)^-(2d+r+2)/2    
    """
    if seed is not None:
        np.random.seed(seed)
    d = 2**n
    rho = np.zeros((d, d), dtype=np.complex128)
    rhos_record = np.zeros((n_iter, d, d), dtype=np.complex128)
    start_time = time.perf_counter()
    for t in range(n_iter):
        
        if t >= n_burnin:
            rho = U @ np.diag(Lamb) @ np.conj(U.T)/(t - n_burnin + 1) + rho*(1-1/(t-n_burnin + 1)) # approximate rho each time as rho_t = gamma_t * V_t * V_t^T /(t-n_burnin) + rho_t-1 / (1 - 1/(t-n_burnin)) -> the later we are, the more importance we give to prev rho
            rhos_record[t, :, :] = rho.copy()
        else:
            rhos_record[t, :, :] = U @ np.diag(Lamb) @ np.conj(U.T)
        cum_times[t] = time.perf_counter() - start_time
    return rhos_record, rho, cum_times


def run_MH_studentt():

    MH_studentt()


if __name__ == "__main__":
    run_MH_studentt()