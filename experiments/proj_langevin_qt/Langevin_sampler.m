function [Y_rho_r_record,t_rec,n_rec] = Langevin_sampler(yhat,As,r,n,N_exp,iter,alpha,lambda,theta,beta,eta,seed)
%
%Tests a projected Langevin algorithm for noisy Quantum Tomography reconstruction 
%
%AJT 18/7/19, based on quantum_tomography_test
%AJT 26/7/22, Binomial noise added
%EM  27/02/23, replaced the gradient descent algorithm by projected
%Langevin sampler 
%EM 1/12/23 : hopefully final implementation
%
% Inputs:
% yhat, As: synthetic data
% r : estimated rank
% n: number of qubits
% N_exp : number of experiments 
% iter: number of gradient iterations
% alpha: weighting parameter of the prior (should be set to 1)
% lambda : weighting parameter for the data fit term: according to theory,
% select ns/2 (with ns the number of shots)
% theta : parameter in the prior
% beta : Langevin noise parameter
% eta : stepsize 
% seed : random seed to replicate experiments


rng(seed);
d = 2^n;

%generate initial point             % same as in Mai and Alquier
M = stiefelcomplexfactory(d,r,1);
V = M.rand();
gamma0 = gamrnd(1/r,1,r,1);
D = diag(gamma0/sum(gamma0));
Y_rho = V*sqrt(D);

% apply the changes of variables described below
Y_rho_r = complextoreal(Y_rho);
As_r = zeros(2*d,2*d,N_exp);
for j = 1:N_exp
    As_r(:,:,j) = complextoreal(As(:,:,j));
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Sanity checks (gradients, Hessians)
% disp('The two following values should be equal if the change of variables is correctly implemented')
% fprintf('Cost in terms of complex variables = %4.2e \n', f_complex(Y_rho,As,yhat,lambda,theta,alpha))  
% fprintf('Cost in terms of real variables = %4.2e \n', f(Y_rho_r,As_r,yhat,lambda,theta,alpha))
%check_gradient_f(Y_rho_r,As_r,yhat,lambda,theta,alpha)   % must display a straight ine in loglog plot with slope -2     


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Start the training  
cost = zeros(1,iter+1);
n_rec = zeros(1,iter+1);
Y_rho_r_record = zeros(2*d,2*r,iter+1);
t_rec = zeros(1,iter);
cost(1) = f(Y_rho_r,As_r,yhat,lambda,theta,alpha);             
k = 1;
Y_rho_r_record(:,:,1) = Y_rho_r;
t_start = tic;

while k<=iter 
    G = gradf(Y_rho_r,As_r,yhat,lambda,theta,alpha);
    n_rec(k) = norm(G,'fro');

    if mod(k-1,1000)==0
        fprintf('Iteration %d, f = %4.2e, norm grad = %4.2e\n', k, cost(k), n_rec(k));
    end

    N1 = randn(d,r);
    N2 = randn(d,r);
    N = [N1, N2; -N2, N1];
    Y_rho_r = Y_rho_r - eta*G+ sqrt(2*eta/beta)*N;                 % Langevin step
    k = k+1;
    cost(k) = f(Y_rho_r,As_r,yhat,lambda,theta,alpha);    
    t_rec(k) = toc(t_start);
    Y_rho_r_record(:,:,k) = Y_rho_r;
end


end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Subfunctions    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% first change of variables
function X_real = complextoreal(X)
    %change of variables to map all complex dxd matrices to real 2dx2d matrices
    %Note that, when X is Hermitian, real(X) is symmetric, 
    %imag(X) is skew-symmetric, so that X_real is real symmetric.
    % scaling is chosen in order to preserve the matrix product
    X_real = [real(X), imag(X); -imag(X), real(X)]*sqrt(2)/2;
end

% inverse change of variables
function X_complex = realtocomplex(X)
    s = size(X);
    X_complex = (X(1:s/2,1:s/2) + 1i*X(1:s/2,s/2+1:end))*sqrt(2);
end


function obj = f_complex(Y_rho,As,yhat,lambda,theta,alpha)
    % f (we are aiming to sample the distribution mu = exp(-f(x))
    % expressed in terms of complex variables
    N_exp = length(yhat);
    [d,r] = size(Y_rho);
    y = zeros(N_exp,1);
    for j = 1:N_exp
        y(j,1) = trace(As(:,:,j)*(Y_rho*Y_rho'));
    end
    obj = lambda*norm(yhat-y)^2 + alpha*(2*d+r+2)*log(det(theta^2*eye(d,d)+Y_rho*Y_rho'))/2;
end


function obj = f(Y_rho_r,As_r,yhat,lambda,theta,alpha)
    % f (we are aiming to sample the distribution mu = exp(-f(x))
    % check in which format we receive the matrix
    N_exp = length(yhat);
    [s1,s2] = size(Y_rho_r);
    d = s1/2; r = s2/2;
    y = zeros(N_exp,1);
    for j = 1:N_exp
        y(j,1) = trace(As_r(:,:,j)*(Y_rho_r*Y_rho_r'));
    end
    obj = lambda*norm(yhat-sqrt(2)*y)^2 + alpha*((2*d+r+2)*log(det(theta^2*eye(2*d,2*d)/sqrt(2)+sqrt(2)*(Y_rho_r*Y_rho_r')))/4 + (2*d+r+2)*d*log(2)/4);
end


% gradient of f
function G = gradf(Y_rho_r,As_r,yhat,lambda,theta,alpha)
    % gradient of f 
    N_exp = length(yhat);
    [s1,s2] = size(Y_rho_r);
    d = s1/2; r = s2/2;
    G = zeros(size(Y_rho_r));
    for j = 1:N_exp
        G = G - 2*sqrt(2)*lambda*(yhat(j)-sqrt(2)*trace(As_r(:,:,j)*(Y_rho_r*Y_rho_r')))*(As_r(:,:,j)+As_r(:,:,j)')*Y_rho_r;
    end
    M = sqrtm(eye(2*r,2*r)+2*(Y_rho_r'*Y_rho_r)/theta^2)\Y_rho_r';
    M = M'*M;
    M(1:d,1:d) = (M(1:d,1:d)+M(d+1:2*d,d+1:2*d))/2;
    M(d+1:2*d,d+1:2*d) = M(1:d,1:d);
    M(1:d,d+1:2*d) = (M(1:d,d+1:2*d)-M(d+1:2*d,1:d))/2;
    M(d+1:2*d,1:d) = -M(1:d,d+1:2*d);
    G = G + alpha*((2*d+r+2)/theta^2*(eye(2*d,2*d)-2*M/theta^2)*Y_rho_r);
    G(1:d,1:r) = (G(1:d,1:r)+G(d+1:2*d,r+1:2*r))/2;
    G(d+1:2*d,r+1:2*r) = G(1:d,1:r);
    G(1:d,r+1:2*r) = (G(1:d,r+1:2*r)-G(d+1:2*d,1:r))/2;
    G(d+1:2*d,1:r) = -G(1:d,r+1:2*r);
end    


function check_gradient_f(Y_rho_r,As_r,yhat,lambda,theta,alpha)
    v = randn(size(Y_rho_r));
    v = v/norm(v,'fro');
    step = logspace(-6,-1,6);
    df = zeros(size(step));
    df2 = zeros(size(step));
    for i = 1:length(step)
        new_Y_rho_r = Y_rho_r+step(i)*v;
        df(i) = (f(new_Y_rho_r,As_r,yhat,lambda,theta,alpha)-f(Y_rho_r,As_r,yhat,lambda,theta,alpha));
        df2(i) = trace(gradf(Y_rho_r,As_r,yhat,lambda,theta,alpha)'*v)*step(i);
    end
    figure;
    loglog(step,abs(df-df2),'-ob');
end


