import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy.stats as stats

import blackjax as bjx

from proj_langevin import f, gradf, complex_to_real, real_to_complex


# def hmc_iter(current_q: np.ndarray, eps: float):
#     """Do one iteration of HMC 
#     """
#     q = current_q




# observed = np.random.normal(10, 20, size=1_000)
# def logdensity_fn(x):
#     logpdf = stats.norm.logpdf(observed, x["loc"], x["scale"])
#     return jnp.sum(logpdf)

# # Build the kernel
# step_size = 1e-3
# inverse_mass_matrix = jnp.array([1., 1.])
# nuts = blackjax.nuts(logdensity_fn, step_size, inverse_mass_matrix)

# # Initialize the state
# initial_position = {"loc": 1., "scale": 2.}
# state = nuts.init(initial_position)

# # Iterate
# rng_key = jax.random.key(0)
# step = jax.jit(nuts.step)
# for _ in range(1_000):
#     rng_key, nuts_key = jax.random.split(rng_key)
#     state, _ = nuts.step(nuts_key, state)




def sample(
    Y_rho0: np.ndarray,
    y_hat: np.ndarray,
    As: np.ndarray,
    r: int,
    n: int,
    n_meas: int,
    n_iter: int,
    n_burnin: int,
):
    """Tentative use of blackjax as a sampling library with the student-t prior
    """
    key = jax.random.key(0)
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







def run_sampler():
    pass


def main():
    run_sampler()

if __name__ == "__main__":
    main()