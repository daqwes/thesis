import numpy as np

s0 = np.eye(2)
sx = np.array([[0, 1], [1, 0]])
sy = np.array([[0, -1j], [1j, 0]])
sz = np.array([[1, 0], [0, -1]])
basis = np.stack([s0, sx, sy, sz])


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
            ret_out[i, :] = arr[i] / np.sqrt((np.abs(arr[i]) ** 2).sum())
        return ret_out
    else:
        return arr / np.sqrt((np.abs(arr) ** 2).sum())


def random_unitary(n: int, p: int, seed=None):
    """Generate a random unitary matrix of size n x p
    Args:
        n (int): number of rows
        p (int): number of columns
    Returns:
        np.ndarray[n, p]
    """
    if seed is not None:
        np.random.seed(seed)
    M_re = np.random.multivariate_normal(np.zeros(n), np.eye(n), size=(p))
    M_im = np.random.multivariate_normal(np.zeros(n), np.eye(n), size=(p))
    M = M_re + M_im * 1j
    Q, _ = np.linalg.qr(M.T, mode="reduced")
    return Q


def gen_true_rho_PL(n: int, rho_type: str|int = "rank1") -> np.ndarray:
    """
    Sample the true matrix rho
    Args:
        rho_type (str|int): type of density matrix we want to sample. Possibilities are
            - rank1
            - rank2
            - approx-rank2
            - rankd
            or an int representing the rank 
    Returns:
        np.ndarray[d, d]: true density matrix
    """
    d = 2**n
    if isinstance(rho_type, (int, np.integer)):
        r = rho_type
        V = random_unitary(d, r)
        gamma = np.random.gamma(1 / r, 1, (r))  # Note: see obs for comment on this
        D = np.diag(gamma) / gamma.sum()
        Y = V @ np.sqrt(D)
        dens_ma = Y @ np.conj(Y.T)
    elif rho_type == "rank1":
        # Pure state - rank1
        r = 1
        V = random_unitary(d, r)
        dens_ma = V @ np.conj(V.T)
    elif rho_type == "rank2":
        # Rank2
        r = 2
        V = random_unitary(d, r)
        v0 = V[:, 0]
        v1 = V[:, 1]
        dens_ma = 1 / 2 * np.outer(v0, np.conj(v0).T) + 1 / 2 * np.outer(
            v1, np.conj(v1.T)
        )
    elif rho_type == "approx-rank2":
        # Approx rank2
        r = 2
        V = random_unitary(d, r)
        v0 = V[:, 0]
        v1 = V[:, 1]
        dens_ma = 1 / 2 * np.outer(v0, np.conj(v0)) + 1 / 2 * np.outer(v1, np.conj(v1))
        w = 0.98
        dens_ma = w * dens_ma + (1 - w) * np.eye(d) / d
    elif rho_type == "rankd":
        # Maximal mixed state (rankd = 16)
        r = d
        V = random_unitary(d, r)
        gamma = np.random.gamma(1 / r, 1, (r))  # Note: see obs for comment on this
        D = np.diag(gamma) / gamma.sum()
        Y = V @ np.sqrt(D)
        dens_ma = Y @ np.conj(Y.T)
    else:
        raise ValueError("rho_type must be one of the possible types")
    return dens_ma

def dec2bin(j: int, n: int):
    """Computes the binary representation of an integer `j` and normalizes it to the length of the binary representation of `n` by adding 0s
    Args:
        j (int): number to convert
        n (int): number of which we take the length to normalize
    Returns:
        binary version j of normalized length (list[int])
    """
    bin_j = bin(j)[2:]
    bin_length = len(bin_j)
    arr = "0" * (2 * n - bin_length) + bin_j
    return [1 if i == "1" else 0 for i in arr]


def pauli_measurements(n: int):
    """Computes the observables/measurements matrices and returns them
    Args:
        n (int): number of qubits
    Returns:
        measurements (np.ndarray)
    """
    As = np.zeros((2**n, 2**n, 4**n), dtype=np.complex128)
    for j in range(4**n):
        bin_ = dec2bin(j, n)
        A = 1
        for i in range(n):
            if bin_[2 * i] == 0 and bin_[2 * i + 1] == 0:
                si = s0
            elif bin_[2 * i] == 0 and bin_[2 * i + 1] == 1:
                si = sx
            elif bin_[2 * i] == 1 and bin_[2 * i + 1] == 0:
                si = sy
            else:
                si = sz
            A = np.kron(si, A)
        As[:, :, j] = A
    return As

def measure_system(As: np.ndarray, rho_true: np.ndarray, n_shots: int|None, n_meas: int) -> np.ndarray:
    """Compute n_shots measurements of the system.
       In case n_shots is None, return the true measurements (expectation of each observable)
    Args:
        As (np.ndarray): observables
        rho_true (np.ndarray): true density matrix
        n_shots (int): number of shots
        n_meas (int): number of measurements/observables
    Returns:
        empirical probabilities (np.ndarray)
    """
    y_hat = np.zeros(n_meas)
    for j in range(n_meas):
        y_hat[j] = min(np.real(np.trace(As[:, :, j] @ rho_true)), 1)
    if n_shots is None:
        return y_hat
    p = (y_hat + 1) / 2

    y_hat = 2 / n_shots * np.random.binomial(n_shots, p) - np.ones(
        y_hat.shape[0]
    )
    return y_hat

def generate_data(n: int, n_meas: int, n_shots: int|None, rho_type: str|int, seed: int|None):
    """Generate a density matrix, and simulate the measurement process
    Args:
        n (int): number of qubits
        n_meas (int): number of measurements/observables to use
        n_shots (int): number of shots
        rho_type (str|int): Type/Rank of density matrix to simulate
        seed (int): optional initial seed
    Returns:
        true rho, observables, empirical probabilities: (tuple[np.ndarray, np.ndarray, np.ndarray])
    """
    if seed is not None:
        np.random.seed(seed)
    d = 2**n
    rho_true = gen_true_rho_PL(n, rho_type)
    As = pauli_measurements(n)

    samples = np.random.choice(range(d * d), n_meas, replace=False)
    As = As[:, :, samples]

    y_hat = measure_system(As, rho_true, n_shots, n_meas)
    return rho_true, As, y_hat

def main():
    n = 3
    d = 2**n
    n_meas = d * d
    n_shots = 2000
    seed = 0
    rho_true, As, y_hat = generate_data(n, n_meas, n_shots, rho_type="rank2", seed=seed)


if __name__ == "__main__":
    main()
