# Efficient sampling algorithms for Bayesian QT

## Problem statement
Quantum tomography (QT) amounts to find a density matrix characterizing the unknown quantum state of a system, given a set of measurements of repeated state preparation.

The quantum tomography problem can be rephrased as a high-dimensional constrained optimization problem, where the sought density matrix is complex-valued, Hermitian, low-rank, and has unit trace.

The dimension of the problem increases exponentially with the number of qubits in the system, so that the development of scalable algorithms for QT is a very active research direction.

A Bayesian formalism can be applied to QT: the set of density matrices is then endowed with a prior distribution, which, combined to a likelihood term accounting for the measurements, leads to a posterior. Sampling this posterior distribution results in an estimator of the sought density matrix together with probabilistic guarantees [1].

[1] T. T. Mai, P. Alquier, Pseudo-Bayesian quantum tomography with rankadaptation, Journal of Statistical Planning and Inference 184, pp. 62-76, 2017. https://doi.org/10.1016/j.jspi.2016.11.003


## Repository structure
The repository is structured as follows:
```
├── data
├── experiments
├── playground
├── README.md
└── src
```
The main folders to look at are `src`, which has all the source files for all the algorithms implementations, and `experiments` which has all the experiments. The `playground` folder is to be ignored as it mostly contains tests.

Each experiment is in its own self-contained folder, with the associated `.py` script to be run, the plots in the `.pdf` format and the output data in the `.csv` file. The experiment folder name hopes to succinctly describe its intent. In case it is not clear, one may look at the `.py` file, where a complete description is available above the function signature.

To run an experiment, simply go into the associated folder and run the script from there:
```bash
python <experiment_name>.py
```


## Installing dependencies

The main dependencies for running all the code are `numpy`, `scipy`, `pandas` and `matplotlib`. If they are not available in your current python setup, you can run
```bash
pip install -r requirements.txt
```

There is also the `requirements_full.txt` file, which also includes `jax` and `blackjax`. The latter provides a lot of samplers out of the box, allowing for an easy way to test their performance for the problem at hand, provided a correct logpdf. These libraries are however not required for the experiments.