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
            ss1 = (Pra_m @ tem_can - p_as)**2  # l^prob: sum_a sum_s (Tr(v P^a_s) - hat(p^_a,s))^2
            ss2 = (Pra_m @ tem - p_as)**2
            ss = (ss1 - ss2).sum()
            # ------------------
            r_prior = (ro-1) * np.log(Te_can[j]/Te[j]) - Te_can[j] + Te[j] # other part of R acceptance ratio
            # ------------------
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


def run_MH_studentt():

    MH_studentt()


if __name__ == "__main__":
    run_MH_studentt()