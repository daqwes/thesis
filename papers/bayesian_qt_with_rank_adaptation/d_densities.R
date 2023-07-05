##########################
# code for distance btw density matrices
##########################
library(gtools)
library(MCMCpack)
library(cmvnorm)
library(ppls)

#Pauli basis 2x2
sx = matrix(c(0,1,1,0),nr=2)
sy = matrix(c(0,1i,-1i,0),nr=2)
sz = matrix(c(1,0,0,-1),nr=2)
basis = list(diag(2),sx,sy,sz)

## #qubits
n = 4
## total number of a, b and r
b = permutations(4,n,  repeats.allowed=T)
a = permutations(3,n, v=c(2,3,4), repeats.allowed=T)
r = permutations(2,n, v = c(-1,1),  repeats.allowed=T)

### Pauli basis for n qubit
library(rTensor)
sig_b = list()
for (i in 1:n^4){
  sig_b[[i]] = kronecker_list(basis[b[i,]])
}

### matrix P_{(r,a);b}
J = 4^n
I = 6^n
d = R = 2^n
A = 3^n
P_rab = matrix(0,nc =J, nr= I)
for(j in 1:J){
  temp = matrix(0,nr=R,nc=A)
  for(s in 1:R){
    for(l in 1:A){
      temp[s,l] = prod(r[s, b[j,]!=1])*prod(a[l,b[j,]!=1]==b[j,b[j,]!=1])
    }
  }
  P_rab[,j] = c(temp)
}

### read data for 4 qubits
W4.Data <- read.delim("~/Documents/thesis/papers/bayesian_qt_with_rank_adaptation/data/W4-Data.dat", header=FALSE)
W4.Data <- as.matrix(W4.Data)
W4.Data <- t(W4.Data[,-1])
p_ra = c(W4.Data)
temp1 = p_ra%*%P_rab
temp1 = temp1/16
# calculating coefficients rho_b
rho_b = c()
for(i in 1:4^n){
  rho_b[i] = temp1[i]/3^(sum(b[i,]==1))
}

### density by inversion
rho.hat = matrix(0,nr=2^n,nc=2^n)
for(s in 1:4^n){
  rho.hat = rho.hat + rho_b[s]*sig_b[[s]]
}

######## Auxiliary functions to speed up ##############
# normalize a complex vector #
norm.complex = function(x) return(x/sqrt(sum(Mod(x)^2)))
# taking the diagonal faster #
Di.ag = function(A,B)return(sapply(1:nrow(A),function(s) A[s,]%*%B[,s]))

### MAIN CODES: ###
rho = matrix(0,nr=2^n,nc=2^n)
Lamb = rdirichlet(1, rep(.1,2^n))
S <- emulator::cprod(rcmvnorm(d,mean=rep(0,d),sigma=diag(d)))
U <- sapply(1:2^n,function(i) norm.complex(rcmvnorm(1,sigma=S)))

gamm = 1/100

RMSE = c()

#MCMC loop
T =100

for( t in 1 : T){
  #update Lambda
  L.can = rdirichlet(1, rep(.1,d))
  sum.L = sum(L.can^2 -Lamb^2) - 2*(L.can -Lamb)%*%Di.ag(U%*%rho.hat,Conj(t(U)))
  ap.L = exp(-sum.L*gamm)
  if(runif(1) <= Re(ap.L)) Lamb <- L.can
  
  #update U
  U.can = sapply(1:d,function(i) norm.complex(rcmvnorm(1,sigma=diag(d))) )
  sum.U = sum(Di.ag((U.can%*%diag(c(Lamb),nrow = d)%*%Conj(t(U.can)) 
                    - U%*%diag(c(Lamb),nrow = d)%*%Conj(t(U))), rho.hat))
  ap.U = exp(-2*gamm*sum.U)
  if(runif(1) <= Re(ap.U) ) U <- U.can
  
  #approx rho
  rho = U%*%diag(c(Lamb),nrow = d)%*%Conj(t(U))/t + rho*(1-1/t)
  RMSE[t] =mean((rho.hat-rho)%*%Conj(t((rho.hat-rho))))
}#end MCMC

#Test for the eigeinvalues
eigen(rho)$values
sum(eigen(rho)$values)
sum(diag(rho))
# plot RMSE
xgr = 1:T
plot(xgr, sqrt(RMSE[xgr]),"l",xlab = "",ylab = "")
RMSE[T]

