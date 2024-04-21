import numpy as np

from proj_langevin import f, gradf, complex_to_real, real_to_complex


def hmc_iter(current_q: np.ndarray, eps: float):
    """Do one iteration of HMC 
    """
    q = current_q

def hmc(
    Y_rho0: np.ndarray,
    y_hat: np.ndarray,
    As: np.ndarray,
    r: int,
    n: int,
    n_meas: int,
    n_iter: int,
    n_burnin: int,
):
    """Tentative implementation of HMC for the student-t prior 
    """
    np.random.seed(0)
    d = 2**n

    Y_rho = Y_rho0
    # Apply change of variable
    Y_rho_r = complex_to_real(Y_rho)

    As_r = np.zeros((2 * d, 2 * d, n_meas))
    for j in range(n_meas):
        As_r[:, :, j] = complex_to_real(As[:, :, j])

    # As_r is real now, no need to use complex dtypes
    As_r_swap = np.empty(As_r.shape[::-1], dtype=np.float64)
    As_r_sum_swap = np.empty((As_r.shape[::-1]),dtype=np.float64)
    for j in range(n_meas):
        As_r_swap[j,:,:] = As_r[:,:,j]
        As_r_sum_swap[j,:,:] = As_r[:, :, j] + np.conj(As_r[:,:,j].T)


    Y_rho_record = np.zeros((n_iter, d, d), dtype=np.complex128)
    t_rec = np.zeros(n_iter)

    for k in range(1, n_iter + 1):
        pass

def run_HMC():
    pass


def main():
    run_HMC()

if __name__ == "__main__":
    main()