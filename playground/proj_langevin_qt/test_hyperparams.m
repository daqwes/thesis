% Main test script
% Author: E. Massart
% Date: Dec. 6, 2023

%Averages over N_rpt seeds Langevin sampling using more and more data: the
%number of experiment replication is given by ns.
% Warning: in the paper, this number is called m, similarly as in Mai and Alquier...

clear all; close all; clc;
N_rpt = 2;
n = 4;
d = 2^n;

ns_i = 2000;
lambda_i = ns_i/2;
eta_i = 0.05/ns_i;
matrix_types = {'rank-one', 'rank-two', 'approx-rank-two', 'full-rank'};
matrix_type = matrix_types{2};  
    
% type should be 'rank-one', 'rank-two', 'approx-rank-two', 'full-rank', 'var-rank'
% if type is 'var-rank', provide the value of m and r as 4th and 5th inputs
iter = 5000;
alpha = 1;
filename = strcat("n_", string(n), "_ns_", string(ns_i), "_lambda_", string(lambda_i), "_eta_", string(eta_i), "_matrix-type_", matrix_type, "_iter_", string(iter), "_alpha_", string(alpha), ".txt");
thetas = [1e-6, 1e-3, 1e-1, 1e0];
betas = [1e1, 1e2, 1e3];
rs = 2:5;
n_rows  = max(d * length(thetas) * length(betas));
results = zeros(n_rows, 4);
row = 1;
for i = 1:length(thetas)
    theta = thetas(i);
    for j = 1:length(betas)
        beta = betas(j);
        for r = rs
            try
                errs = zeros(N_rpt, 1);
                for k = 1:N_rpt
                    seed = k;
                    [yhat,As,rho_true,N_exp] = problem_gen(matrix_type,n,ns_i,seed);
                    [Y_rho_r_record,t_rec,norm_rec] = Langevin_sampler(yhat,As,r,n,N_exp,iter,alpha,lambda_i,theta,beta,eta_i,seed);
                    M_avg = mean(Y_rho_r_record(:,:,end-100:end),3);
                    M_avg = sqrt(2)*(M_avg*M_avg');
                    M_avg = (M_avg(1:d,1:d) + 1i*M_avg(1:d,d+1:end))*sqrt(2);
                    err = norm(M_avg-rho_true,'fro')^2;

                    errs(k, 1) = err;
                end
                avg_errs = mean(errs, 1);
                results(row, 1) = theta;
                results(row, 2) = beta;
                results(row, 3) = r;
                results(row, 4) = avg_errs;
            catch e
                results(row, 1) = theta;
                results(row, 2) = beta;
                results(row, 3) = r;
                results(row, 4) = -1;
            end
            row = row + 1;
        end
    end
end

writematrix(results, filename)