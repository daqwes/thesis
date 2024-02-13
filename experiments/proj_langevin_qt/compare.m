% Main test script
% Author: E. Massart
% Date: Dec. 6, 2023

%Averages over N_rpt seeds Langevin sampling using more and more data: the
%number of experiment replication is given by ns.
% Warning: in the paper, this number is called m, similarly as in Mai and Alquier...

clear all; close all; clc;
N_rpt = 5;
% seed = 1:N_rpt;
seed = 0;
% N = 20; 
% ns = ceil(logspace(1,7,N));    % number of replications of the numerical experiments
% lambda = ns./2;           % parameter scaling the likelihood compared to the prior
% eta = 0.1./ns;             % stepsize (should depend on lambda to avoid exploding gradients)
err_avg_rec = zeros(4,N_rpt);


n = 3;
d = 2^n;

ns_i = 5000;
lambda_i = ns_i/2;
eta_i = 0.05/ns_i;
matrix_types = {'rank-one', 'rank-two', 'approx-rank-two', 'full-rank'};
matrix_type = matrix_types{2};  
    
% type should be 'rank-one', 'rank-two', 'approx-rank-two', 'full-rank', 'var-rank'
% if type is 'var-rank', provide the value of m and r as 4th and 5th inputs
iter = 5000;
alpha = 1;

theta = 0.1;              % parameter inside the prior (multiplying the identity, which is added to the density matrix to make it full rank)
beta = 100;               % noise parameter in Langevin
r = 6;

[yhat,As,rho_true,N_exp] = problem_gen(matrix_type,n,ns_i,seed);
% yhat
% quit
[Y_rho_r_record,t_rec,norm_rec] = Langevin_sampler(yhat,As,r,n,N_exp,iter,alpha,lambda_i,theta,beta,eta_i,seed);
M_avg = mean(Y_rho_r_record(:,:,end-100:end),3);
M_avg = sqrt(2)*(M_avg*M_avg');
M_avg = (M_avg(1:d,1:d) + 1i*M_avg(1:d,d+1:end))*sqrt(2);
err = norm(M_avg-rho_true,'fro');
err^2