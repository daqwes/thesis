import numpy as np
import h5py

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


def random_complex_ortho(n: int, p: int):
    """Generate a random orthonormal matrix of size n x p
    Args:
        n (int): number of rows
        p (int): number of columns
    Returns:
        np.ndarray[n, p]
    """
    M_re = np.random.multivariate_normal(np.zeros(n), np.eye(n), size=(p))
    M_im = np.random.multivariate_normal(np.zeros(n), np.eye(n), size=(p))
    M = M_re + M_im * 1j
    Q, _ = np.linalg.qr(M.T, mode="reduced")
    return Q


def gen_true_rho_PL(n: int, rho_type: str = "rank1") -> np.ndarray:
    """Sample true rho
    Args:
        rho_type (str): type of density matrix we want to sample. Possibilities are
            - rank1
            - rank2
            - approx-rank2
            - rankd
    Returns:
        np.ndarray[d, d]: true density matrix
    """
    d = 2**n
    if rho_type == "rank1":
        # Pure state - rank1
        r = 1
        V = random_complex_ortho(d, r)
        dens_ma = V @ np.conj(V.T)
    elif rho_type == "rank2":
        # Rank2
        r = 2
        V = random_complex_ortho(d, r)
        v0 = V[:, 0]
        v1 = V[:, 1]
        dens_ma = 1 / 2 * np.outer(v0, np.conj(v0).T) + 1 / 2 * np.outer(
            v1, np.conj(v1.T)
        )
    elif rho_type == "approx-rank2":
        # Approx rank2
        r = 2
        V = random_complex_ortho(d, r)
        v0 = V[:, 0]
        v1 = V[:, 1]
        dens_ma = 1 / 2 * np.outer(v0, np.conj(v0)) + 1 / 2 * np.outer(v1, np.conj(v1))
        w = 0.98
        dens_ma = w * dens_ma + (1 - w) * np.eye(d) / d
    elif rho_type == "rankd":
        # Maximal mixed state (rankd = 16)
        r = d
        V = random_complex_ortho(d, r)
        gamma = np.random.gamma(1 / r, 1, (r))  # Note: see obs for comment on this
        D = np.diag(gamma) / gamma.sum()
        Y = V @ np.sqrt(D)
        dens_ma = Y @ np.conj(Y.T)
    else:
        raise ValueError("rho_type must be one of the possible types")
    return dens_ma


def gen_true_rho_bqst(n: int, rho_type: str = "rank1") -> np.ndarray:
    """Sample true rho
    Args:
        rho_type (str): type of density matrix we want to sample. Possibilities are
            - rank1
            - rank2
            - approx-rank2
            - rankd
    Returns:
        np.ndarray[d, d]: true density matrix
    """
    d = 2**n
    if rho_type == "rank1":
        # Pure state - rank1
        dens_ma = np.zeros((d, d), dtype="complex")
        dens_ma[0, 0] = 1
    elif rho_type == "rank2":
        # Rank2
        v1 = np.zeros(d, dtype="complex")
        v1[0 : int(d / 2)] = 1
        v1 = norm_complex(v1)
        v2 = np.zeros(d, dtype="complex")
        v2[int(d / 2) : d] = 1j
        v2 = norm_complex(v2)
        dens_ma = 1 / 2 * np.outer(v1, np.conj(v1)) + 1 / 2 * np.outer(v2, np.conj(v2))
    elif rho_type == "approx-rank2":
        # Approx rank2
        v1 = np.zeros(d, dtype="complex")
        v1[0 : int(d / 2)] = 1
        v1 = norm_complex(v1)
        v2 = np.zeros(d, dtype="complex")
        v2[int(d / 2) : d] = 1j
        v2 = norm_complex(v2)
        dens_ma = 1 / 2 * np.outer(v1, np.conj(v1)) + 1 / 2 * np.outer(v2, np.conj(v2))
        w = 0.98
        dens_ma = w * dens_ma + (1 - w) * np.eye(d) / d
    elif rho_type == "rankd":
        # Maximal mixed state (rankd = 16)
        u = norm_complex(
            np.random.multivariate_normal(
                np.zeros(d * 2), np.eye(d * 2) / 100, size=(d)
            ).view(np.complex128)
        )
        dens_ma = np.conj(u.T) @ u / d
    else:
        raise ValueError("rho_type must be one of the possible types")
    return dens_ma


def dec2bin(j: int, n: int):
    bin_j = bin(j)[2:]
    bin_length = len(bin_j)
    arr = "0" * (2 * n - bin_length) + bin_j
    return [1 if i == "1" else 0 for i in arr]


def pauli_measurements(n: int):
    """
    Args:
        n (int): number of qubits
    Returns:
        np.ndarray
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
            # if j == 2:
                # print(si)
        # print(A)
        # if j == 9:
        #     exit(1)
        # print(nnz[1] + A.shape[1]*nnz[0] +1)
        As[:, :, j] = A
    return As


def dump_h5(data: np.ndarray, var_name: str):
    data_d = np.array(data).astype(np.complex128).transpose().copy()#.reshape(*data.shape[::-1]).copy()#  .reshape(*data.shape[::-1])
    data_d2 = np.array(data).astype(np.complex128).reshape(data.shape[::-1]).copy()
    imag_data_d = np.where(np.imag(data_d) == 0, 0, np.imag(data_d))
    real_data_d = np.where(np.real(data_d) == 0, 0, np.real(data_d))
    # for j in range(data.shape[-1]):
    #     print(np.real(data[:,:,j]))

    with h5py.File(f"{var_name}_py.h5", "w") as file:
        file.create_dataset("data_real", data=real_data_d)
        file.create_dataset("data_imag", data=imag_data_d)


def generate_data(n: int, n_exp: int, n_shots: int, rho_type: str):
    """Generate a density matrix, and simulate the measurement process
    Args:
        n (int): number of qubits
        rho_type (str): Type of density matrix to simulate
    Returns:

    """
    np.random.seed(0)
    d = 2**n
    rho_true = gen_true_rho_PL(n, rho_type)
    As = pauli_measurements(n)
    # As = As[:, :, :n_exp]

    samples = np.random.choice(range(d * d), n_exp, replace=False)
    As = As[:, :, samples]
    y_hat = np.zeros(n_exp)
    for j in range(n_exp):
        y_hat[j] = min(np.real(np.trace(As[:, :, j] @ rho_true)), 1)
    p = (y_hat + 1) / 2
    # TODO make sure this is the correct approach, and if it's correct to obtain
    # values in the [-1, 1] interval 
    y_hat = 2 / n_shots * np.random.binomial(n_shots, p) - np.ones(
        y_hat.shape[0]
    )
    return rho_true, As, y_hat



def main():
    n = 3
    d = 2**n
    n_exp = d * d
    n_shots = 2000
    seed = 0
    rho_true, As, y_hat = generate_data(n, n_exp, n_shots, rho_type="rank2")
    # dump_h5(As, "As")
    # a = np.array(range(1,25)).reshape(4, 2, 3)
    # print(a)
    # print(np.ravel(a, "K"))
    # dump_h5(a, "range")
    # print(rho_true)
    # print(As)
    # print(y_hat)


if __name__ == "__main__":
    main()
