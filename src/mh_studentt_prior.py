import time
import sys
import numpy as np
from scipy.special import binom

parent_module = sys.modules['.'.join(__name__.split('.')[:-1]) or '__main__']
if __name__ == '__main__' or parent_module.__name__ == '__main__':
    from data_generation_exact import random_multivariate_complex, random_standard_exponential, random_uniform, generate_data_exact_PL
    from data_generation import norm_complex
    from proj_langevin import gen_init_point, complex_to_real, real_to_complex
else:
    from .data_generation_exact import random_multivariate_complex, random_standard_exponential, random_uniform, generate_data_exact_PL
    from .data_generation import norm_complex
    from .proj_langevin import gen_init_point, complex_to_real, real_to_complex

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
        # print(lik, prior)
        post = -(lik + prior)
    # post = exp(-f) with f = L + log(P) -> exp(-L)/P
    else:
        lik = np.exp(
            - lambda_ * np.linalg.norm(y_hat - np.sqrt(2) * y) ** 2
        )
        prior = np.linalg.det(theta**2 * np.eye(2 * d) / np.sqrt(2)
                    + np.sqrt(2) * Y_rho_r_outer) ** ((2*d + r + 2)/4)
        # print(lik, prior)
        post = lik/prior
    return post
    
def eval_proposal(Y_next: np.ndarray, Y_prev: np.ndarray, dist: str = "normal", scaling_coef: float = 1.0):
    # TODO
    m, n = Y_next.shape
    if dist == "normal":
        return 1/(2 * np.pi)**(m*n) * np.exp(-1/2 * np.linalg.norm(Y_next, ord="fro")**2)
    elif dist == "normal_dep":
        # return 1/(2 * np.pi)**(m*n) * np.exp(-1/2 * np.linalg.norm(Y_next - Y_prev, ord="fro")**2)
        return 1 # symmetric proposal
    elif dist == "goe":
        assert m == n, "d and r must have the same value for GOE"
        Z = (4 * np.pi)**(m/2) * (2*np.pi)**(1/2 * binom(m, 2))
        return np.exp(-1/4 * np.trace(Y_next @ Y_next)) / Z
    elif dist == "exp_dep":
        return 1 # not symmetric proposal (and 1/x is not a valid pdf, but used in Mai/Alquier)
    elif dist == "unitary_rank1":
        return 1 # unsure on how to compute it here
    else:
        raise ValueError("dist is not a valid type")
def sample_proposal(d: int, r: int, Y_curr: np.ndarray, seed: int, dist: str = "normal", scaling_coef: float = 1.0):
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
        std = scaling_coef
        Y_prop = std * np.random.randn(d, r)
        return Y_prop
    elif dist == "normal_dep":
        std = scaling_coef
        Y_prop = std * np.random.randn(d, r)
        return Y_prop + Y_curr
    elif dist == "exp_dep":
        # Similar to what is done in Mai & Alquier 2017
        # This will return values which are very close to 1 (exp(-0,5) = 0.6 and exp(0.5) = 1.6)
        param = scaling_coef
        rd = np.exp(param * np.random.uniform(-0.5, 0.5, Y_curr.shape))
        Y_prop = Y_curr * rd
        return Y_prop
    elif dist == "goe":
        assert d == r, "d and r must have the same value for GOE"
        G = np.random.randn(d, r)
        return (G+G.T)/(np.sqrt(2))
    elif dist == "unitary_rank1":
        Y_prop = gen_init_point(d, r, seed)
        return Y_prop#/Y_prop.sum()
    else:
        raise ValueError("dist type not valid, provide correct value")
def acc_ratio(Y_next: np.ndarray, Y_prev: np.ndarray, As: np.ndarray, As_r_swap: np.ndarray, y_hat: np.ndarray, lambda_: float, theta: float, prop_dist: str = "normal", scaling_coef_prop: float = 1.0, use_prop_in_ratio: bool = False, log_transform: bool = False) -> float:
    """
    r = prop(x|x') * post(x') / prop(x'|x) * post(x)
    Here, post = exp(-(L + log(prior))).
    Then r = prop(x|x') * exp(-(L(x')  + prior(x')))/ prop(x'|x) * exp(-(L(x) + log(prior(x))))
    log(r) = log(prop(x|x')) -(L(x') + prior(x')) - (log(prop(x'|x)) -(L(x) + log(prior(x)))
           = log(prop(x|x')) - log(prop(x'|x)) + L(x) + log(prior(x)) - L(x') - prior(x')
           = log(prop(x|x')) - log(prop(x'|x)) + log(post(x')) - log(post(x))
    """
    if log_transform:
        prop = np.log(eval_proposal(Y_prev, Y_next, prop_dist, scaling_coef_prop)) - np.log(eval_proposal(Y_next, Y_prev, prop_dist, scaling_coef_prop))
        post = eval_posterior_real(Y_next, As_r_swap, y_hat, lambda_, theta, log_transform=True)\
               - eval_posterior_real(Y_prev, As_r_swap, y_hat, lambda_, theta, log_transform=True)
        if use_prop_in_ratio:
            ratio = prop + post
        else:
            ratio = post # ignore the proposal
    else:
        n_prop = eval_proposal(Y_prev, Y_next, prop_dist, scaling_coef_prop)
        d_prop = eval_proposal(Y_next, Y_prev, prop_dist, scaling_coef_prop) 
        n_post = eval_posterior_real(Y_next, As_r_swap, y_hat, lambda_, theta, log_transform)
        d_post = eval_posterior_real(Y_prev, As_r_swap, y_hat, lambda_, theta, log_transform) 
        if use_prop_in_ratio:
            n = n_post * n_prop
            d = d_post * d_prop
        else:
            # Not taking into account the proposal
            n = n_post #* n_prop
            d = d_post #* d_prop
        # print(eval_proposal(next, prop_dist), eval_posterior_real(prev, As_r_swap, y_hat, lambda_, theta ))
        # print(eval_proposal(prev, prop_dist), eval_posterior_real(next, As_r_swap, y_hat, lambda_, theta))
        ratio = n/d
        print(n_post, d_post, ratio)
        if np.isnan(ratio):
            raise ValueError("Ratio is incorrect, div by 0")
    return min(1, ratio)


def MH_studentt(n: int, y_hat: np.ndarray, As: np.ndarray, Y0: np.ndarray, lambda_: float, theta: float, seed: int, n_iter: int = 500, n_burnin: int = 100, proposal_dist: str = "exp_dep", scaling_coef_prop: float = 1, use_prop_in_ratio: bool = False, log_transform: bool= True) -> np.ndarray:
    """ Metropolis-Hastings algorithm with a student-t prior   
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
    acc_count = 0
    for t in range(n_iter):
        # Shape should be (2d, 2r) as we now work in the real domain
        Y_r_next = sample_proposal(int(2*d), int(2*r), Y_r_prev, seed=None, dist=proposal_dist, scaling_coef=scaling_coef_prop)
        rd = np.random.random()
        if log_transform:
            rd = np.log(rd)
        ratio = acc_ratio(Y_r_next, Y_r_prev, As, As_r_swap, y_hat, lambda_, theta, proposal_dist, scaling_coef_prop, use_prop_in_ratio, log_transform=log_transform)
        # print(rd, ratio)
        if rd < ratio:
            Y_r_prev = Y_r_next
            acc_count+=1
        rho_iter = np.sqrt(2) * (Y_r_prev @ np.conj(Y_r_prev.T))
        rhos_record[t, :, :] = real_to_complex(rho_iter)
        cum_times[t] = time.perf_counter() - start_time

    acc_rate = acc_count / n_iter
    print(f"Acceptance rate: {acc_rate}")
    return rhos_record, rho_iter, cum_times, acc_rate 


def run_MH_studentt(n: int, n_shots: int, As: np.ndarray, y_hat: np.ndarray, n_iter: int = 500, n_burnin: int = 100, seed: int = None, run_avg: bool = True, proposal_dist: str = "exp_dep", scaling_coef_prop: float = 1.0, use_prop_in_ratio: bool = False, log_transform: bool = True, init_point: np.ndarray|None = None):
    if seed is not None:
        np.random.seed(seed)
    d = 2**n
    if init_point is not None:
        _, r = init_point.shape
        Y0 = init_point
    else:
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
    rhos_record, rho_mh_stt, cum_times, acc_rate = MH_studentt(n, y_hat, As, Y0, lambda_, theta, seed, n_iter, n_burnin, proposal_dist, scaling_coef_prop, use_prop_in_ratio, log_transform)
    
    if run_avg:
        M_avg = np.zeros_like(rhos_record[0,:,:])
        for i in range(n_iter - n_burnin):
            M_avg = rhos_record[n_burnin + i,:,:] * 1/(i + 1) + M_avg  * (1 - 1 /(i+1))
            # print((M_avg - Y_rho_record[n_burnin + i]).abs().sum())
            rhos_record[n_burnin + i] = M_avg.copy()
    else:
        M_avg = np.mean(rhos_record[n_burnin:,:,:], 0) # 

    return rhos_record, M_avg, cum_times, acc_rate

def main():
    seed = 0
    n = 3
    n_exp = 3**n
    n_shots = 2000
    n_iter = 5000
    n_burnin = 1000
    run_avg = True
    log_transform = True
    proposal = "unitary_rank1"
    scaling_coef_prop = 1
    use_prop_in_ratio = False
    rho_true, As, y_hat = generate_data_exact_PL(n, n_exp, n_shots, rho_type="rank2", seed=seed)
    np.random.seed()
    rhos_mh_stt, rho_mh_stt, cum_times_mh_stt, acc_rate  = run_MH_studentt(n, n_shots, As, y_hat, n_iter, n_burnin, seed = None, run_avg=run_avg, proposal_dist=proposal, scaling_coef_prop = scaling_coef_prop, use_prop_in_ratio = use_prop_in_ratio, log_transform=log_transform)
    # print(rho_true)
    # print(rho_mh_stt)
    err = np.linalg.norm(rho_mh_stt - rho_true, "fro")
    print(err**2)

if __name__ == "__main__":
    main()