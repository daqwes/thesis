# Efficient sampling algorithms for Bayesian QT

Quantum tomography (QT) amounts to find a density matrix characterizing the unknown quantum state of a system, given a set of measurements of repeated state preparation.

The quantum tomography problem can be rephrased as a high-dimensional constrained optimization problem, where the sought density matrix is complex-valued, Hermitian, low-rank, and has unit trace.

The dimension of the problem increases exponentially with the number of qubits in the system, so that the development of scalable algorithms for QT is a very active research direction.

A Bayesian formalism can be applied to QT: the set of density matrices is then endowed with a prior distribution, which, combined to a likelihood term accounting for the measurements, leads to a posterior. Sampling this posterior distribution results in an estimator of the sought density matrix together with probabilistic guarantees [1].

[1] T. T. Mai, P. Alquier, Pseudo-Bayesian quantum tomography with rankadaptation, Journal of Statistical Planning and Inference 184, pp. 62-76, 2017. https://doi.org/10.1016/j.jspi.2016.11.003
