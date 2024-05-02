# src directory

There are 4 main files that are of interest here:

- `data_generation.py`: Includes all functions related to the approximated data generation process. The driver function in this file is `generate_data`.
- `data_generation_sep.py`: Includes all functions related to the separate generation process. The driver function in this file is `generate_data_exact`.
- `metropolis_hastings.py`: Includes all functions related to the prob estimator. The driver function in this file is `run_MH`. The name may change in the future to better distinguish itself from other approaches.
- `proj_langevin.py`: Includes all functions related to the projected langevin method. The driver function in this file is `run_PL`.