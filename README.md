# Numerical comparison of MCMC methods for Quantum Tomography

## Author: Daniel Mokeev

This work was done in the context of a Master's thesis at Université Catholique de Louvain. It was supervised by Estelle Massart and Tameem Adel.
## Abstract
Quantum Tomography is a process to reconstruct the state of a quantum system. By measuring replicas of the state, we can estimate the density matrix that represents it. Many methods exist to approximate the density matrix, including direct and optimization based approaches. In recent years however, Bayesian methods have emerged as a promising alternative thanks to their ability to incorporate prior information and quantify uncertainty. In this work, our contribution is twofold. First, we numerically compare 2 recent MCMC methods, the prob-estimator and the Projected Langevin algorithm, in different experimental setups. Second, we introduce 2 new algorithms which combine the prior used in Projected Langevin with the algorithm from the prob-estimator. This allows us to evaluate the advantages that a gradient-based method brings, as well as the impact of a Student-t prior on the result.


## Repository structure
The repository is structured as follows:
```
├── experiments
├── README.md
└── src
```

The `src` directory contains all the source files for all the algorithms implementations, and `experiments` which has all the experiments. See this [file](experiments/README.md) which describes how to run the experiments, as well as which experiment connects to which plot/table.

## Installing dependencies

The main dependencies for running all the code are `numpy`, `scipy`, `pandas` and `matplotlib`. If they are not available in your current python setup, you can run
```bash
pip install -r requirements.txt
```
The code was tested with `python3.12`, however other versions might also work.