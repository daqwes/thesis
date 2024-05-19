import numpy as np
from numpy.linalg import eig
import itertools
import functools
from typing import *
import sys

parent_module = sys.modules['.'.join(__name__.split('.')[:-1]) or '__main__']
if __name__ == '__main__' or parent_module.__name__ == '__main__':
    from data_generation import norm_complex
else:
    from .data_generation import norm_complex

sx = np.array([[0, 1], [1, 0]])
sy = np.array([[0, 1j], [-1j, 0]])
sz = np.array([[1, 0], [0, -1]])
basis = np.stack([np.eye(2), sx, sy, sz])


def projectors_py(idx_list, r_):
    """
    Returns the P^{a_i}_{s_i} list of projection matrices
    Note1: the evs returned by numpy and the ones obtained don't match, 
    but as they are not unique, it is still correct.
    """
    r_idx = [1 if i==-1 else 0 for i in r_]
    evs = [eig(basis[i])[1] for i in idx_list]
    selected_evs = np.array([ev[:, r_idx[i]] for i, ev in enumerate(evs)])
    ret = np.array([np.outer(np.conj(ev), ev) for ev in selected_evs])
    return ret

def random_uniform(low: float, high: float, size: tuple[int, ...], seed: int|None = None) -> np.ndarray | float: 
    if seed is not None:
        np.random.seed(seed)
    if isinstance(size, int) and size == 1:
        return np.random.uniform(low, high)
    else:
        return np.random.uniform(low, high, size)

def random_multivariate_complex(mean: np.ndarray, cov: np.ndarray, size: tuple[int, ...], seed: int|None = None) -> np.ndarray:
    if seed is not None:
        np.random.seed(seed)
    if isinstance(size, int) and size == 1:
        return np.random.multivariate_normal(mean, cov) + 1j * np.random.multivariate_normal(mean, cov)
    else:
        return np.random.multivariate_normal(mean, cov, size) + 1j * np.random.multivariate_normal(mean, cov, size)

def random_standard_exponential(size: tuple[int, ...], seed: int|None = None) -> np.ndarray | float:
    if seed is not None:
        np.random.seed(seed)
    if isinstance(size, int) and size == 1:
        return np.random.standard_exponential()
    else:
        return np.random.standard_exponential(size) 

def compute_rho_inversion(n: int, b: np.ndarray, p_as: np.ndarray, P_rab: np.ndarray, sig_b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the rho estimate with the inversion method
    Args:
        p_as (np.ndarray[R*A]): Vector mapping each observable and result combination to its empirical probability
        P_rab: (np.ndarray[I=2^n x 3^n, J=4^n])
        sig_b: (np.ndarray[J, d=16, d])
    Returns:
        Tuple[np.ndarray[d=2^n, d], np.ndarray]: the approximation of rho using the inversion technique, and its eigenvectors 
    """
    d = 2**n
    J = 4**n
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

def get_observables(n: int):
    A = 3**n
    R = 2**n
    a = np.array(list(itertools.product(range(1, 4), repeat=n))) # {x,y,z}^n
    r = np.array(list(itertools.product([-1, 1], repeat=n)))
    Pra = []
    for j in range(A):
        for i in range(R):
            Pra.append(np.array(functools.reduce(np.kron, projectors_py(a[j], r[i]))).flatten("F"))
    Pra = np.array(Pra)
    return Pra

def get_observables_PL_format(n: int):
    """
    Returns observables in the [d, d, n_obs: R1 R2 R3..] format
    """
    d = 2**n
    A = 3**n
    R = 2**n
    a = np.array(list(itertools.product(range(1, 4), repeat=n))) # {x,y,z}^n
    r = np.array(list(itertools.product([-1, 1], repeat=n)))
    Pra = np.zeros((d, d, R*A), dtype=np.complex128)
    for j in range(A):
        for i in range(R):
            idx = i + j*R
            Pra[:,:,idx] = functools.reduce(np.kron, projectors_py(a[j], r[i]))
    return Pra

# def init_matrices(n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
#     """Initializes all the matrices for the simulation of the data, the inversion method and for the prob-estimator
#     Returns:
#         Tuple[np.ndarray[A*R, d*d], np.ndarray[J, d*d], np.ndarray[I, J]]: Pra, sig_b, P_rab
#     """
#     A = 3**n
#     R = 2**n
#     J = 4**n
#     I = 6**n
#     b = np.array(list(itertools.product(range(4), repeat=n))) # {I, x, y, z}^n
#     a = np.array(list(itertools.product(range(1, 4), repeat=n))) # {x,y,z}^n
#     r = np.array(list(itertools.product([-1, 1], repeat=n)))

#     # Corresponds to P^a_s in paper (each row here is a matrix), size: 2^n x 3^n flattened
#     Pra = []
#     for j in range(A):
#         for i in range(R):
#             Pra.append(np.array(functools.reduce(np.kron, projectors_py(a[j], r[i]))).flatten("F"))
#     Pra = np.array(Pra)
    
#     sig_b = np.array([functools.reduce(np.kron, (basis[b[i,:], :, :])) for i in range(J)])
#     P_rab = np.zeros((I, J))
#     for j in range(J):
#         tmp = np.zeros((R, A))
#         for s in range(R):
#             for l in range(A):
#                 val = np.prod(r[s, b[j] != 0])\
#                     * np.prod(a[l, b[j] != 0] == b[j, b[j]!=0])
#                 tmp[s,l] = val
#         P_rab[:, j] = tmp.flatten(order="F")

#     return Pra, sig_b, P_rab, b, a, r

def get_true_rho(n: int, rho_type: str = "rank1", seed=None) -> np.ndarray:
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
    if seed is not None:
        np.random.seed(seed)
    d = 2**n
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



def compute_measurements(n: int, dens_ma, n_shots: int | None=2000, seed=None):
    """Simulate the data measurement process for the prob estimator
    Args:
        dens_ma (np.ndarray[d= 2^n, d]): True rho
        n_shots (int): Number of measurements
    Returns:
        np.ndarray[R*A]: Vector mapping each observable and result combination to its empirical probability
    """
    if seed is not None:
        np.random.seed(seed)
    A = 3**n
    R = 2**n
    a = np.array(list(itertools.product(range(1, 4), repeat=n))) # {x,y,z}^n
    r = np.array(list(itertools.product([-1, 1], repeat=n)))
    # Corresponds to Tr(rho \dot P^a_s), which in turn corresponds to p_a,s
    # These probabilities are the true ones, so we do not have access to them
    # We use them below to measure the system
    Prob_ar = np.zeros((A, R), dtype=np.complex128)
    if n==1:
        for i in range(A):
            for j in range(R):
                Prob_ar[i,j] = dens_ma.flatten(order="F") @ projectors_py(a[i], r[j])
    else:
        for i in range(A):
            for j in range(R):
                Prob_ar[i,j] = np.diag(dens_ma @ np.array(functools.reduce(np.kron, projectors_py(a[i], r[j])))).sum()
    Prob_ar = np.real(Prob_ar)
    if n_shots is None:
        return Prob_ar.flatten(order="F")
    # For each observable, we sample n_shots samples 
    # according to the true probabilities calculated above, and then give the probability
    # For example: 
    # if n=4 (qubits) and a_i = {x, x, y, z}, then an outcome could be s_j {-1, 1, -1, 1}
    # For this pair of a,s, we calculate the number of times we sampled this situation (the H == s part) 
    # and get the empirical probability for this combination

    p_ra = np.zeros((R, A)) # = \hat{p}_a,s
    for i, x in enumerate(Prob_ar):
        H = np.random.choice(R, n_shots, replace=True, p=x) #n_shots elements
        out = []
        for s in range(R):
            out.append((H==s).sum()/n_shots) # Calculate the empirical prob of this combination
        p_ra[:, i] = out
    # Transform matrix to vector form
    return p_ra.flatten(order="F")


def generate_data_sep(n: int, n_meas: int, n_shots: int|None, rho_type: str, seed: int|None):
    """Generate a density matrix, and simulate the measurement process
    Args:
        n (int): number of qubits
        rho_type (str): Type of density matrix to simulate
    Returns:

    """
    d = 2**n
    if seed is not None:
        np.random.seed(seed)
    A = 3**n
    R = 2**n
    # Here we need to select the observables randomly, based on samples.
    # Because the maximum number of pauli matrices combinations is A = 3**n (for which we then select the 2**n possible combinations of possible output)
    # we sample n_meas among A, and then select its R = 2**n associated outcomes, requiring the range select 
    samples = np.random.choice(A, n_meas, replace=False) * R
    samples_ranges = [list(range(i, i+R)) for i in samples]
    Pra = get_observables(n)
    Pra = Pra[samples_ranges, :].reshape(n_meas * R, -1)
    rho_true = get_true_rho(n, rho_type, seed=None)

    # Return size of 2**n x 3**n, flattened by col ([R1 R2 R3]) 
    y_hat = compute_measurements(n, rho_true, n_shots, seed=None)
    y_hat = y_hat[samples_ranges].reshape(-1)
    return rho_true, Pra, y_hat


def generate_data_sep_PL(n: int, n_meas: int, n_shots: int|None, rho_type: str, seed: int|None):
    """Generate a density matrix, and simulate the measurement process. Returns the measure
    Args:
        n (int): number of qubits
        rho_type (str): Type of density matrix to simulate
    Returns:

    """
    d = 2**n
    A = 3**n
    R = 2**n
    if seed is not None:
        np.random.seed(seed)
    # See explanation above, in generate_data_sep
    samples = np.random.choice(A, n_meas, replace=False) * R
    samples_ranges = [list(range(i, i+R)) for i in samples]
    Pra = get_observables_PL_format(n)
    Pra = Pra[:,:, samples_ranges].reshape(d, d, -1)
    rho_true = get_true_rho(n, rho_type, seed=None)

    y_hat = compute_measurements(n, rho_true, n_shots, seed=None)
    y_hat = y_hat[samples_ranges].reshape(-1)
    return rho_true, Pra, y_hat