function [yhat,As,rho_true,N_exp] = problem_gen(type,n,ns,seed,varargin)

% Inputs: 
%   type: string capturing the type of input data.
%       Can be: 'rank-one', 'rank-two', 'approx_rank_two', 'full-rank', 'var-rank'
%   n: number of qubits
%   ns: number of shots (repeat measurements)
%   varargin 1 (only used if type == 'var-rank') : N_exp: number of measurements (must be <= 4^n)
%       If type != 'var-rank': we assume the complete measurement setting    
%   varargin 2 (only used if type == 'var-rank') : r: upper bound on the rank of the quantum state

% Outputs: 
%   yhat: measured frequencies (corrupted by binomial noise) 
%   As: measurement operators
%   rho_true: true density
%   N_exp : number of measurements

d = 2^n;
rng(seed);

if strcmp('rank-one',type)
    % complete measurement setting (thus m = d^2), pure state, ns trials
    % per experiment
    r = 1;
    N_exp = d^2;

    %initialise Stiefel manifold object
    M = stiefelcomplexfactory(d,r,1);
    V = M.rand();
    rho_true = V*V';
    
elseif strcmp('rank-two',type)

    % complete measurement setting (thus m = d^2), pure state, ns trials
    % per experiment
    r = 2;
    N_exp = d^2;

    %initialise Stiefel manifold object
    M = stiefelcomplexfactory(d,r,1);
    V = M.rand();
    rho_true = 0.5*V(:,1)*V(:,1)'+0.5*V(:,2)*V(:,2)';
    

elseif strcmp('approx-rank-two',type)
    r = 2;
    N_exp = d^2;

    %initialise Stiefel manifold object
    M = stiefelcomplexfactory(d,r,1);
    V = M.rand();
    rho_true = 0.98*(0.5*V(:,1)*V(:,1)'+0.5*V(:,2)*V(:,2)')+0.02*eye(d)/d;


elseif strcmp('full-rank',type)
    % complete measurement setting (thus m = d^2), pure state, ns trials
    % per experiment
    r = d;
    N_exp = d^2;


    %initialise Stiefel manifold object
    M = stiefelcomplexfactory(d,r,1);
    V = M.rand();
    gamma = gamrnd(1/r,1,r,1);  % gamma distribution wish shape parameters 1/r and r
    D = diag(gamma/sum(gamma));
    Y = V*sqrt(D);
    rho_true = Y*Y';
    

elseif strcmp('var-rank',type)

    N_exp = varargin{1};
    r = varargin{2};

    %initialise Stiefel manifold object
    M = stiefelcomplexfactory(d,r,1);
    V = M.rand();
    gamma = gamrnd(1/r,1,r,1);  % gamma distribution wish shape parameters 1/r and r
    D = diag(gamma/sum(gamma));
    Y = V*sqrt(D);
    rho_true = Y*Y';
        
end


%generate random Pauli matrix measurements
As = pauli_measurements(n);
samples = randsample(d^2,N_exp);   % sample m entries uniformly at random from 1:d^2, without replacement
As = As(:,:,samples);
for j = 1:N_exp
    yhat(j,1) = min(real(trace(As(:,:,j)*rho_true)),1);
end
p = (yhat+1)/2;   
yhat = 2/ns*binornd(ns, p)-ones(size(yhat,1),1);       %binomial distribution with parameters ns (number of trials) and p (probability of success)

end