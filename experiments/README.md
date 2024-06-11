# Experiments

This folder contains all the experiments that were done in the context of this thesis.

Each experiment is in its own self-contained folder, with the associated `.py` script to be run, the plots in the `.pdf` format and the output data in the `.csv` file. The experiment folder name hopes to succinctly describe its intent. In case it is not clear, one may look at the `.py` file, where a complete description is available above the function signature.

To run an experiment, simply go into the associated folder and run the script from there:
```bash
python <experiment_name>.py
```

## Experiments described in paper

Not all experiments were kept for the final version of the thesis. Here is a reference which maps the figure/table numbers to the original files:
### Chapter 4
#### Baseline convergence
- Figure 4.1(a): [`iterations_no_avg`](iterations_no_avg/iters_acc_comp_iters_no_avg.pdf)
- Figure 4.1(b):  [`iterations_no_avg_sep`](iterations_no_avg_sep/iters_acc_comp_iters_no_avg_sep.pdf)
#### Convergence across ranks
- Figure 4.2(a): [`iterations_no_avg_rank1`](iterations_no_avg_rank1/iters_acc_comp_iters_no_avg_rank1.pdf)
- Figure 4.2(b): [`iterations_no_avg`](iterations_no_avg/iters_acc_comp_iters_no_avg.pdf)
- Figure 4.2(c):
[`iterations_no_avg_rankd`](iterations_no_avg_rankd/iters_acc_comp_iters_no_avg_rankd.pdf)
#### Convergence across qubit counts
- Figure 4.3(a): [`iterations_no_avg`](iterations_no_avg/iters_acc_comp_iters_no_avg)
- Figure 4.3(b):
[`iterations_no_avg_n4`](iterations_no_avg_n4/iters_acc_comp_iters_no_avg_n4.pdf)
- Figure 4.3(c):[`iterations_no_avg_n5`](iterations_no_avg_n5/iters_acc_comp_iters_no_avg_n5.pdf)
- Figure 4.4(a): [`iterations_no_avg`](iterations_no_avg/iters_acc_comp_time_no_avg)
- Figure 4.4(b):
[`iterations_no_avg_n4`](iterations_no_avg_n4/iters_acc_comp_time_no_avg_n4.pdf)
- Figure 4.4(c):[`iterations_no_avg_n5`](iterations_no_avg_n5/iters_acc_comp_time_no_avg_n5.pdf)
#### Impact of burn-in duration
- Figure 4.5(a): [`burnin_no_avg`](burnin_no_avg/burnin_acc_comp_burnin.pdf)
- Figure 4.5(b): [`burnin_no_avg_sep`](burnin_no_avg_sep/burnin_acc_comp_burnin_sep.pdf)
#### Impact of number of shots
- Figure 4.6(a): [`shots`](shots/shots_acc_comp_shots_exp_loglog.pdf)
- Figure 4.6(b): [`shots_sep`](shots_sep/shots_acc_comp_shots_exp_sep_loglog.pdf)
- Figure 4.7: [`shots_lambda_prob_with_var_lambda`](shots_lambda_prob_with_var_lambda/shots_acc_comp_shots_exp_lambda_prob_with_var_lambda_loglog.pdf)
#### Impact of number of measurements
- Figure 4.8(a): [`meas`](meas/meas_acc_comp_meas.pdf)
- Figure 4.8(b): [`meas_sep`](meas_sep/meas_acc_comp_meas_sep.pdf)
#### Impact of the knowledge of the rank
- Figure 4.9(a): [`rank_known`](rank_known/rank_known.pdf)
- Figure 4.9(b): [`rank_not_known`](rank_not_known/rank_not_known.pdf)
- Figure 4.10(a): [`rank_known_var_theta_pl`](rank_known_var_theta_pl/rank_known_var_theta_pl.pdf)
- Figure 4.10(b): [`rank_not_known_var_theta_pl`](rank_not_known_var_theta_pl/rank_not_known_var_theta_pl.pdf)
- Figure 4.11(a): [`rank_known_n4`](rank_known_n4/rank_known_n4.pdf)
- Figure 4.11(b): [`rank_not_known`](rank_not_known_n4/rank_not_known_n4.pdf)
### Chapter 5
- Table 5.2: [`prop_search_mhgs`](prop_search_mh_studentt_run_avg/prop_search_mh_studentt_run_avg.csv)
- Table 5.4: [`prop_search_mhgs`](prop_search_mhgs/prop_search_mhgs.csv)
- Figure 5.1: [`iterations_no_avg_sep_prob_pl_mhs_mhgs`](iterations_no_avg_sep_prob_pl_mhs_mhgs/iters_acc_comp_iters_no_avg_sep_prob_pl_mhs_mhgs.pdf)
