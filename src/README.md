# src directory

There are 6 main files that are of interest here:

- `data_generation.py`: Includes all functions related to the approximated data generation process. The driver function in this file is `generate_data`.
- `data_generation_sep.py`: Includes all functions related to the separate generation process. The driver function in this file is `generate_data_sep`.
- `metropolis_hastings.py`: Includes all functions related to the prob estimator. The driver function in this file is `run_MH`.
- `proj_langevin.py`: Includes all functions related to the projected langevin method. The driver function in this file is `run_PL`.
- `mh_studentt_prior.py`: Includes all functions related to the MHS method. The driver function in this file is `run_MH_studentt`.
- `mh_gibbs_studentt_prior.py`: Includes all functions related to the MHGS method. The driver function in this file is `run_MH_gibbs_studentt`.
