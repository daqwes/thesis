##########################
# code for distance btw probability matrices
##########################
library(gtools)
library(MCMCpack)
library(cmvnorm)
library(rTensor)
library(doSNOW)
library(ppls)
######## Auxiliary functions to speed up ##############
# normalize a complex vector #
norm.complex = function(x) return(x/sqrt(sum(Mod(x)^2)))
# taking the diagonal faster #
Di.ag = function(A,B)return(sapply(1:nrow(A),function(s) A[s,]%*%B[,s]))
# return the eigen vectors
eigen.vec = function(A) return(eigen(A)$vectors)
# return the projectors
projectors = function(a,r){
   tem1 = lapply(basis[a],eigen.vec)
   tem3 = lapply(1:length(tem1), function(s) return(tem1[[s]][r[s],]))
   lapply(tem3, function(x) x%*%Conj(t(x)))
}

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
r = permutations(2,n, v = c(1,2),  repeats.allowed=T)

### Pauli basis for n qubit
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

# The "assume-true" density matrix
density = matrix(0,nr=d,nc=d)
density[1,1] = 1

# the probabilities matrix of the r.vs P.ar
Prob.ar = matrix(0,nr=A,nc=R)
for(i in 1:A){
  for(j in 1:R){
    Prob.ar[i,j] <- sum(Di.ag(density,kronecker_list(projectors(a = a[i,],r = r[j,]))))
  }
}
Prob.ar = Re(Prob.ar)

# calculate the probability matrix after simulating data
mp_ra = apply(Prob.ar, 1,function(x){
  H = sample(1:R,20,prob = x,replace=TRUE)
  return(sapply(1:R,function(s)sum(H==s)/100))
})
p_ra = c(mp_ra)
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

### MAIN CODES: ###
rho = matrix(0,nr=2^n,nc=2^n)
Lamb = rdirichlet(1, rep(.1,2^n))
S <- emulator::cprod(rcmvnorm(d,mean=rep(0,d),sigma=diag(d)))
U <- sapply(1:2^n,function(i) norm.complex(rcmvnorm(1,sigma=S)))

gamm = 1/2

RMSE = c()

#MCMC loop
T =10000

for( t in 1 : T){
   #update Lambda
   L.can = rdirichlet(1, rep(.1,d))
   temL = 0
   for(rj in 1:R){
    temL = sum(sapply(1:A,function(ai)c(U%*%diag(c(L.can+Lamb))%*%Conj(t(U)))%*%c(kronecker_list(projectors(a = a[ai,],r = r[rj,]))-2*t(mp_ra)[ai,rj])*c(U%*%diag(c(L.can-Lamb))%*%Conj(t(U)))%*%c(kronecker_list(projectors(a = a[ai,],r = r[rj,])))   
                      )) + temL
   }
   ap.L = exp(-gamm*Re(temL))
   if(runif(1) <= ap.L) Lamb <- L.can
   
   #update U
#   U.can = sapply(1:d,function(i) norm.complex(rcmvnorm(1,sigma=diag(d))) )
#   temU = foreach(ai=1:A)%:%foreach(rj=1:R,.packages='rTensor')%dopar%{
#      (c(U%*%diag(c(L.can+Lamb))%*%Conj(t(U)))%*%c(
#        kronecker_list(projectors(a = a[ai,],r = r[rj,])))
#       -2*W4.Data[ai,rj])*sum(Di.ag(U.can%*%diag(c(Lamb))%*%Conj(t(U.can)) 
#                +U%*%diag(c(Lamb))%*%Conj(t(U)),
#                kronecker_list(projectors(a = a[ai,],r = r[rj,]))))
#   }
#   ap.U = exp(-gamm*sum(unlist(temU)))
#   if(runif(1) <= Re(ap.U)) U <- U.can
   
   #approx rho
   rho = U%*%diag(c(Lamb),nrow = d)%*%Conj(t(U))/t + rho*(1-1/t)
   RMSE[t] =  mean(Mod(rho-rho.hat)^2)
}#end MCMC

foreach(i=1:A)%:%foreach(j=1:R,.packages='rTensor')%dopar%{
   kronecker_list(list(matrix(1,nr=2,nc=2),matrix(2,nc=2,nr=2)))}



     