from data_generation_exact import generate_data_exact

import numpy as np

import matplotlib.pyplot as plt

def main():

    n = 2
    n_exp = 8
    d = 2**n
    # rho_true, Pra, y_hat = generate_data_exact(n, n_exp, 2000, "rank2", 0)
    # print(Pra)

    # print(y_hat.shape)
    np.random.seed(0)

    A = 5 # 3**n
    R = 3 # 2**n
    dd = 4 # d*d
    n_exp = 3
    As = np.random.random((A * R , dd))
    y_hat = np.random.random(R*A)
    samples = np.random.choice(A, n_exp, replace=False) * R
    samples_ranges = [list(range(i, i+R)) for i in samples]
    As_s = As[samples_ranges, :].reshape(n_exp * R, -1)
    y_hat_s = y_hat[samples_ranges].reshape(-1)
    print(As)
    print(y_hat)
    print(samples)
    print(As_s.shape, As_s)
    print(y_hat_s.shape, y_hat_s)


def sample_test():
    n_samples = 10000
    samples = [0] *n_samples
    samples_01 = [0] *n_samples
    for i in range(n_samples):
        v = np.random.uniform(-0.5, 0.5)
        v2 = np.random.uniform(0, 1)
        samples[i] = np.exp(v)
        samples_01[i] = np.exp(v2)
    plt.hist(samples, density=True)
    plt.hist(samples_01, density=True)
    x = np.arange(0.6, 3, 0.1)
    y = 1/x
    plt.plot(x,y)
    plt.show()
if __name__ == "__main__":
    sample_test()