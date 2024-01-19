% Main test script
% Author: E. Massart
% Date: Dec. 6, 2023

%Averages over N_rpt seeds Langevin sampling using more and more data: the
%number of experiment replication is given by ns.
% Warning: in the paper, this number is called m, similarly as in Mai and Alquier...

clear all; close all; clc;
N_rpt = 5;
seed = 1:N_rpt;
N = 20; 
ns = ceil(logspace(1,7,N));    % number of replications of the numerical experiments
lambda = ns./2;           % parameter scaling the likelihood compared to the prior
eta = 0.1./ns;             % stepsize (should depend on lambda to avoid exploding gradients)
err_avg_rec = zeros(N,N_rpt);


n = 3;
d = 2^n;

for i = 1:N
    
    fprintf('--------------------------------------------  ns = %d \n', ns(i));
    
    for j = 1:N_rpt
    
        % type should be 'rank-one', 'rank-two', 'approx-rank-two', 'full-rank', 'var-rank'
        % if type is 'var-rank', provide the value of m and r as 4th and 5th inputs
        iter = 5000;
        alpha = 1;
        theta = 1;              % parameter inside the prior (multiplying the identity, which is added to the density matrix to make it full rank)
        beta = 1e2;               % noise parameter in Langevin
        r = 5;

        [yhat,As,rho_true,N_exp] = problem_gen('rank-two',n,ns(i),seed(j));

        [Y_rho_r_record,t_rec,norm_rec] = Langevin_sampler(yhat,As,r,n,N_exp,iter,alpha,lambda(i),theta,beta,eta(i),seed(j));

        M_avg = mean(Y_rho_r_record(:,:,end-100:end),3);
        M_avg = sqrt(2)*(M_avg*M_avg');
        M_avg = (M_avg(1:d,1:d) + 1i*M_avg(1:d,d+1:end))*sqrt(2);
        err_avg_rec(i,j) = norm(M_avg-rho_true,'fro');

    end

end
    
record_param = struct;
record_param.n = n;
record_param.d = d;
record_param.r = r;
record_param.ns = ns;
record_param.iter = iter;
record_param.alpha = alpha;
record_param.lambda = lambda;
record_param.theta = theta;
record_param.beta = beta;
record_param.eta = eta;
record_param.seed = seed;
record_param.err_avg_rec = err_avg_rec;

s = string(datetime);
indx = strfind(s,' ');
s{1}(indx) = '_';
indx = strfind(s,':');
s{1}(indx) = '_';
save(strcat('res_sampling_',s,'.mat'),'record_param','-v7.3');

figure;
err_to_plot = mean(err_avg_rec.^2,2);
loglog(ns,err_to_plot,'.-b');
xlabel('$m$','Interpreter','LaTex','Fontsize',15);
ylabel('$||\hat \rho-\rho_0||^2_F$','Interpreter','LaTex','Fontsize',15);
