import numpy as np
import time 
import sys

parent_module = sys.modules['.'.join(__name__.split('.')[:-1]) or '__main__']
if __name__ == '__main__' or parent_module.__name__ == '__main__':
    from data_generation import generate_data, random_unitary
    from data_generation_sep import init_matrices, get_true_rho, compute_measurements, compute_rho_inversion, random_uniform, random_multivariate_complex, get_observables, random_standard_exponential, projectors_py, generate_data_sep
    from proj_langevin import gen_init_point
    from utils import compute_error
else:
    from .data_generation import generate_data, random_unitary
    from .data_generation_sep import init_matrices, get_true_rho, compute_measurements, compute_rho_inversion, random_uniform, random_multivariate_complex, get_observables, random_standard_exponential, projectors_py, generate_data_sep
    from .proj_langevin import gen_init_point
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

def MH_prob(n: int, p_as: np.ndarray, Pra_m: np.ndarray, u_hat: np.ndarray, gamm: float, pkl: str|None, seed: int|None, n_iter: int = 500, n_burnin: int = 100) -> np.ndarray:
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
    rhos_record[0,:,:] = u_hat
    start_time = time.perf_counter()
    acc_count_gamma = 0
    acc_count_V = 0
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
            r_prior = (ro-1) * np.log(Te_can[j]/Te[j]) - Te_can[j] + Te[j] # other part of R acceptance ratio
            ap = -gamm*np.real(ss) # other part (why use np.real?)
            if np.log(random_uniform(0, 1, 1, seed=seed)) <= ap + r_prior: 
                Te = Te_can # if value above draw from U(0,1), then update
                acc_count_gamma +=1
            Lamb = Te/Te.sum() # gamma
        for j in range(d): # Loop for V_i
            U_can = U.copy()
            # Removed: replace RW by IMH
            # rd_U = U[:,j] + random_multivariate_complex(np.zeros(d), np.eye(d), 1, seed)/100#np.random.multivariate_normal(np.zeros(d*2),np.eye(d*2)/100, size=(1)).view(np.complex128)
            rd_U = random_multivariate_complex(np.zeros(d), np.eye(d), 1, seed)/100
            U_can[:, j] = norm_complex(rd_U) # Sample U/V from the unit sphere (not sure why we add to previous value)
            tem_can = (U_can @ np.diag(Lamb) @ np.conj(U_can.T)).flatten(order="F") # gamma * U * U^T
            tem = (U @ np.diag(Lamb) @ np.conj(U.T)).flatten(order="F") # gamma * U_t-1 * U^T_t-1
            ss1 = (Pra_m @ tem_can - p_as)**2
            ss2 = (Pra_m @ tem - p_as)**2
            ss = (ss1 - ss2).sum()
            ap = -gamm * np.real(ss) # other part of A accep ratio
            if np.log(random_uniform(0, 1, 1, seed)) <= ap: 
                U = U_can # if value above draw from U(0,1), then update
                acc_count_V += 1
        if t < n_iter - 1:
            if t >= n_burnin:
                rho = U @ np.diag(Lamb) @ np.conj(U.T)/(t - n_burnin + 1) + rho*(1-1/(t-n_burnin + 1)) # approximate rho each time as rho_t = gamma_t * V_t * V_t^T /(t-n_burnin) + rho_t-1 / (1 - 1/(t-n_burnin)) -> the later we are, the more importance we give to prev rho
                rhos_record[t+1, :, :] = rho.copy()
            else:
                rhos_record[t+1, :, :] = U @ np.diag(Lamb) @ np.conj(U.T)
        cum_times[t] = time.perf_counter() - start_time

    ac_rate_gamma = acc_count_gamma / (n_iter * d)
    ac_rate_V = acc_count_V / (n_iter * d)
    # print(ac_rate_gamma, ac_rate_V) 
    return rhos_record, rho, cum_times

def run_MH(n: int, n_meas: int, n_shots: int, rho_true: np.ndarray, As: np.ndarray, y_hat: np.ndarray, n_iter: int = 500, n_burnin: int = 100, seed: int|None|None = 0, init_point: np.ndarray|None = None, gamma: float|None = None):
    """Runner function for the prob-estimator
    Args:
        n (int): number of qubits
        n_meas (int): number of measurements
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

    if gamma is not None:
        gamm = gamma
    else:
        gamm = n_shots/2 

    d = 2**n
    if init_point is not None:
        u_hat = init_point
    else:
        u_hat = gen_init_point(d, d) 
    rhos_record, rho_prob, cum_times = MH_prob(n, y_hat, As, u_hat, gamm, None, seed = None, n_iter=n_iter, n_burnin=n_burnin)
    return rhos_record, rho_prob, cum_times

def main():
    seed = 0
    n = 3
    d = 2**n
    n_meas = d * d
    n_shots = 100000
    rho_type = "rank2"
    rho_true, As, y_hat = generate_data(n, n_meas, n_shots, rho_type=rho_type)

    As_flat = np.zeros((4**n, 2**n * 2**n), dtype = np.complex128)
    for i in range(4**n):
        # TODO: it is not clear why this works better than `flatten(order="F")`
        # as it is more correct to use the latter (similar to what is done in R)
        As_flat[i,:] = As[:,:,i].flatten(order="C") 
    
    # rho_hat, u_hat = compute_rho_inversion(n, y_hat)

    gamm = n_shots/2 # lambda in paper
    n_iter = 2000
    n_burnin = 100
    u_hat = random_unitary(d, d)
    rho_prob = MH_prob(n, y_hat, As_flat, u_hat, gamm, None, seed, n_iter, n_burnin)
    err = np.linalg.norm(rho_prob- rho_true, "fro")
    print(err**2)
    # print(rho_true[:4, :4])
    # print(rho_prob[:4, :4])
    # print(y_hat.shape)

def main_sep_data_gen():
    seed = 1
    n = 3
    d = 2**n
    n_meas = 3**n
    n_iter = 600
    n_burnin = 100
    n_shots = 2000
    gamm = n_shots/2

    rho_type = "rank2" 

    # Pra = get_observables(n)
    # rho_true = get_true_rho(n, rho_type, seed=seed)
    # u_hat = random_unitary(d, d, seed=seed)
    # y_hat = compute_measurements(n, rho_true, n_shots, seed=None)

    rho_true, As, y_hat = generate_data_sep(n, n_meas, n_shots, rho_type=rho_type, seed=seed)

    init_point = random_unitary(d,d)# gen_init_point(d, d)

    n_samples = 10
    avg_err = 0
    avg_err_mse = 0
    for i in range(n_samples):
        rhos_record, rho_prob, cum_times = run_MH(n, n_meas, n_shots, rho_true, As, y_hat, n_iter, n_burnin, seed=None, init_point=init_point, gamma=gamm)#MH_prob(n, y_hat, As, u_hat, gamm, None, seed=prob_seed, n_iter=n_iter, n_burnin=n_burnin)
        err_mse = np.real(compute_error(rho_prob, rho_true, "MSE"))
        err_fro_sq = compute_error(rho_prob, rho_true, "fro_sq")
        avg_err += err_fro_sq
        avg_err_mse += err_mse 
        print(err_mse, err_fro_sq)
    avg_err /= n_samples
    avg_err_mse /= n_samples
    print(f"MSE: {err_mse}")
    print(f"Fro^2: {avg_err}")

    
if __name__ == "__main__":
    main_sep_data_gen()