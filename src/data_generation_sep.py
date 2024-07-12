import numpy as np
from numpy.linalg import eig
import itertools
import functools
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
    Calculates and returns the P^{a_i}_{s_i} list of projector matrices
    Args:
        idx_list (list[int]): indices of which basis to select 
        r_ (list[int]): indices of which eigenvectors to select
    Returns:
        projector matrices (np.ndarray)
    """
    r_idx = [1 if i==-1 else 0 for i in r_]
    evs = [eig(basis[i])[1] for i in idx_list]
    selected_evs = np.array([ev[:, r_idx[i]] for i, ev in enumerate(evs)])
    ret = np.array([np.outer(np.conj(ev), ev) for ev in selected_evs])
    return ret

def random_uniform(low: float, high: float, size: tuple[int, ...]|int, seed: int|None = None) -> np.ndarray | float: 
    """
    Returns a random uniform value or a numpy array of dimension `size` of values. Each of them is iid and comprised between `low` and `high`.
    """
    if seed is not None:
        np.random.seed(seed)
    if isinstance(size, int) and size == 1:
        return np.random.uniform(low, high)
    else:
        return np.random.uniform(low, high, size)

def random_multivariate_complex(mean: np.ndarray, cov: np.ndarray, size: tuple[int, ...]|int, seed: int|None = None) -> np.ndarray:
    """
    Returns a random complex vector or tensor of dimension `size` from a multivariate normal(`mean`,`cov`). Each column is iid.
    """
    if seed is not None:
        np.random.seed(seed)
    if isinstance(size, int) and size == 1:
        return np.random.multivariate_normal(mean, cov) + 1j * np.random.multivariate_normal(mean, cov)
    else:
        return np.random.multivariate_normal(mean, cov, size) + 1j * np.random.multivariate_normal(mean, cov, size)

def random_standard_exponential(size: tuple[int, ...], seed: int|None = None) -> np.ndarray | float:
    """
    Returns a random value from the standard exponential distribution or a tensor of dimension `size` of values. Each entry is iid.
    """
    if seed is not None:
        np.random.seed(seed)
    if isinstance(size, int) and size == 1:
        return np.random.standard_exponential()
    else:
        return np.random.standard_exponential(size) 

def get_observables(n: int):
    """
    Calculates and returns the measurements/projectors for the separate qubit method for `n` qubits.
    The format is [R1=d x d flattened, R2, R3, ..] = 6**n x (d x d)
    """
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
    Returns projectors in the [d, d, n_obs: R1 R2 R3..] format for `n` qubits
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

def get_true_rho(n: int, rho_type: str = "rank1", seed=None) -> np.ndarray:
    """ 
    Sample the true matrix rho 
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
        # Maximal mixed state (rankd = 16 if n==4)
        u = norm_complex(np.random.multivariate_normal(np.zeros(d*2),np.eye(d*2)/100, size=(d)).view(np.complex128))
        dens_ma = np.conj(u.T) @ u /d
    else:
        raise ValueError("rho_type must be one of the possible types")
    return dens_ma



def compute_measurements(n: int, dens_ma, n_shots: int | None=2000, seed=None):
    """Simulate the data measurement process for the prob estimator
    Args:
        n (int): number of qubits
        dens_ma (np.ndarray[d= 2^n, d]): True rho
        n_shots (int): Number of measurements
        seed (int): optional initial seed
    Returns:
        np.ndarray[R*A]: Vector mapping each projector and result combination to its empirical probability
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
    # For each projector, we sample n_shots samples 
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
    """Generate a density matrix, and simulate the measurement process for the separate qubit process
    Args:
        n (int): number of qubits
        n_meas (int): number of measurements/projectors
        n_shots (int): number of shots to perform
        rho_type (str): Type of density matrix to simulate
        seed (int): potential initial seed
    Returns:
        true rho, measurements/projectors, empirical probabilities associated to projectors (tuple[np.ndarray, np.ndarray, np.ndarray])
    """
    if seed is not None:
        np.random.seed(seed)
    A = 3**n
    R = 2**n
    # Here we need to select the projectors randomly, based on samples.
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
    """Generate a density matrix, and simulate the measurement process. Returns the measurements/projectors in the PL format
    Args:
        n (int): number of qubits
        n_meas (int): number of measurements/projectors
        n_shots (int): number of shots
        rho_type (str): Type of density matrix to simulate
        seed (int): optional initial seed
    Returns:
        true rho, projectors/measurements, empirical probabilities (tuple[np.ndarray, np.ndarray, np.ndarray])
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