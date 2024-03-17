library(reticulate)
# use_python("/home/daniel/micromamba/envs/mcmc/bin/python")
use_condaenv("road_segmentation")
# normalizePath()
##########################
# code for distance btw probability matrices
##########################
library(gtools)

library(cmvnorm)
library(rTensor)
library(ppls)
library(rhdf5)

######## Auxiliary functions to speed up ##############
# normalize a complex vector #
norm.complex = function(x) return(x/sqrt(sum(Mod(x)^2)))
norm.complex <- compiler::cmpfun(norm.complex)
# taking the diagonal faster #
Di.ag = function(A,B)return(sapply(1:nrow(A),function(s) A[s,]%*%B[,s]))
Di.ag <- compiler::cmpfun(Di.ag)
# return the eigen vectors
eigen.vec = function(A) return(eigen(A)$vectors)


#Pauli basis 2x2
sx = matrix(c(0,1,1,0),nr=2)
sy = matrix(c(0,1i,-1i,0),nr=2)
sz = matrix(c(1,0,0,-1),nr=2)
basis = list(diag(2),sx,sy,sz)

## #qubits
n = 3
J = 4^n
I = 6^n
d = R = 2^n
A = 3^n
## total number of a, b and r
b = permutations(4,n,  repeats.allowed=T)
a = permutations(3,n, v=c(2,3,4), repeats.allowed=T)
r = permutations(2,n, v = c(-1,1),  repeats.allowed=T)

# return the projectors
projectors = function(a,r){
   tem1 = lapply(basis[a],eigen.vec)
   tem3 = lapply(1:length(tem1), function(s) return(tem1[[s]][r[s],]))
   # print(tem3)
   # print(tem3[[1]] %*%Conj(t(tem3[[1]])))
   # print("\n")
   lapply(tem3, function(x) x%*%Conj(t(x)))
}

### Pauli basis for n qubit
sig_b = list()
for (i in 1:J){
   sig_b[[i]] = kronecker_list(basis[b[i,]])
}

### matrix P_{(r,a);b}
P_rab = matrix(0,nc =J, nr= I)
for(j in 1:J){
   temp = matrix(0,nr=R,nc=A)
   for(s in 1:R){
      for(l in 1:A){
         val = prod(r[s, b[j,]!=1]) * prod(a[l,b[j,]!=1]==b[j,b[j,]!=1])
         # print(b[j,b[j,]!=1])
         temp[s, l] = val
      }
   }
   # print(temp[1:3, 1:20])
   # print(c(temp))
   # if (j == 5) break()
   P_rab[,j] = c(temp)
}

as.integer_neg0 <- function(x) ifelse(x == 0, 0, as.integer(x))
as.matrix_fromlist <- function(x) {
   num_matrices <- length(x)
   rows <- nrow(x[[1]])
   cols <- ncol(x[[1]])
   dim2 <- rows * cols
   if (is.null(rows) || is.null(cols)){
      dim2 <- length(x[[1]])
   }
   # Combine the matrices into a 2D array
   conv <- array(unlist(x), dim = c(num_matrices, dim2))
   class(conv) <- "numeric"
   conv
}

# h5write(as.integer_neg0(P_rab), "P_rab_r.h5", "data")
# h5write(as.matrix_fromlist(sig_b), "sig_b_r.h5", "data")

# The projectors matrices
Pra = list()
count = 0
for(j in 1:A){
   for(i in 1:R){
      count = count + 1
      # print(a[j,])
      # print(r[i,])
      # print(projectors(a =a[j,],r = r[i,]))
      # print(" ")
      Pra[[count]] = c(kronecker_list(projectors(a =a[j,],r = r[i,])))
   }
}

# h5write(as.matrix_fromlist(Pra), "Pra_r.h5", "data")

# The "true-test" dens.ma matrix
## pure state
# dens.ma = matrix(0,nr=d,nc=d)
# dens.ma[1,1] = 1
## mixed state
u = sapply(1:d, function(i)norm.complex(rcmvnorm(1,sigma=diag(d)/100)))
dens.ma = Conj(t(u))%*%u/d
# v1 = norm.complex(rcmvnorm(1,sigma=diag(d)))
# v2 = norm.complex(rcmvnorm(1,sigma=diag(d)))
# v1 = t(rep(0,d))
# v1[1:(d/2)]=1
# v1 = norm.complex(v1)
# v2 = t(rep(0,d))
# v2[d:(d/2+1)] = 1i
# v2 = norm.complex(v2)
# dens.ma <- Conj(t(v1))%*%v1*0.5 + Conj(t(v2))%*%v2*0.5
# dens.ma = Conj(t(v1))%*%v1*0.4999 + Conj(t(v2))%*%v2*0.4999+(1-2*0.4999)*diag(d)/d

# the probabilities matrix of the P.ar, used to simulate data
Prob.ar = matrix(0,nr=A,nc=R)
if(n==1){
   for(i in 1:A){
      for(j in 1:R){
         Prob.ar[i,j] <- c(dens.ma)%*%unlist(projectors(a = a[i,],r = r[j,]))
      }  }
}else{
   for(i in 1:A){
      for(j in 1:R){
         Prob.ar[i,j] <- sum(Di.ag(dens.ma,kronecker_list(projectors(a = a[i,],r = r[j,]))))
      }}}
Prob.ar = Re(Prob.ar)

# calculate the probability matrix after simulating sample
n.size = 2000## numbers of repeat the measurements
p_ra = apply(Prob.ar, 1,function(x){
   H = sample(1:R,n.size,prob = x,replace=TRUE)
   return(sapply(1:R,function(s)sum(H==s)/n.size))
})

# transform the matrix to the vector form
p_ra1 = c(p_ra)  
temp1 = p_ra1%*%P_rab
temp1 = temp1/d

# calculating coefficients rho_b
rho_b = c()
for(i in 1:J){
   rho_b[i] = temp1[i]/3^(sum(b[i,]==1))
}

### density by inversion
rho.hat = matrix(0,nr=d,nc=d)
for(s in 1:J){
   rho.hat = rho.hat + rho_b[s]*sig_b[[s]]
}
u.hat = eigen(rho.hat)$vectors
### renormalize lambda.hat
lamb.til = eigen(rho.hat)$value
lamb.til[which(lamb.til<0)] <-0
lamb.hat = lamb.til/sum(lamb.til)

################################################
####### MAIN CODES: ############################
################################################
rho = matrix(0,nr=d,nc=d)
Te = rexp(d)
Id = diag(d)
import(data)
# U <- u.hat
U <- rn
Lamb = c(Te/sum(Te))
ro = .5
S = (rho.hat+Conj(t(rho.hat)))/2
Id = diag(d)
be = 1

gamm = n.size/2
entry = c()
#MCMC loop
Iter = 500
burnin = 100

start_time <- Sys.time()
for(t in 1:(Iter+burnin)){
   print(t)
   for(j in 1:d){
      #update Lambda
      Te.can = Te
      Te.can[j] = Te[j]*exp(be*runif(1,min=-0.5,.5))
      L.can = Te.can/sum(Te.can)
      tem.can = c(U%*%diag(L.can)%*%Conj(t(U)))
      tem = c(U%*%diag(Lamb)%*%Conj(t(U)))
      # Computes the inner product between nu and each combination of (flattened) projectors (results in size 1296, ie 2^n*3^n)
      # and then computes the differences with p_ra1
      ss = sum((sapply(Pra, function(x)crossprod(tem.can,x)) -p_ra1)^2-
                  (sapply(Pra, function(x)crossprod(tem,x)) -p_ra1)^2)
      r.prior = (ro-1)*log(Te.can[j]/Te[j]) -Te.can[j] +Te[j]
      ap = -gamm*Re(ss)
      if(log(runif(1)) <= ap + r.prior){Te<- Te.can}
      Lamb = c(Te/sum(Te))
   }
   #update U
   for(j in 1:d){
      U.can = U
      U.can[,j] = norm.complex(U[,j]+rcmvnorm(1,sigma=diag(d))/100)
      tem.can =c(U.can%*%diag(Lamb)%*%Conj(t(U.can)))
      tem = c(U%*%diag(Lamb)%*%Conj(t(U)))
      ss = sum((sapply(Pra, function(x)crossprod(tem.can,x)) -p_ra1)^2-
                  (sapply(Pra, function(x)crossprod(tem,x)) -p_ra1)^2)
      ap = Re(-gamm*ss)
      if(log(runif(1)) <= ap){U<- U.can}
   }
   #approx rho
   if(t>burnin){
      rho = U%*%diag(Lamb)%*%Conj(t(U))/(t-burnin) + rho*(1-1/(t-burnin))
   #   print(max(Lamb))
   }
}
end_time <- Sys.time()
end_time - start_time
# mean((dens.ma-rho)%*%Conj(t((dens.ma-rho)))) #0.0002687214
tr((rho - dens.ma) %*% Conj(t(rho - dens.ma)))/(d*d) # MSE
tr((rho - dens.ma) %*% Conj(t(rho - dens.ma))) # SSE = fro^2
# mean((rho.hat-dens.ma)%*%Conj(t((rho.hat-dens.ma))))
# eigen(rho,only.values = T)
# sum(diag(rho))
# plot(eigen(rho,only.values = T)$values)
# plot(eigen(dens.ma,only.values = T)$values)
# abline(h=0.1)

# For n = 3:
# For 2000 shots:
# 1e-3
# 1e-1
# For 10000 shots:
# 1e-3
# 1e-1



# For n = 4:
# For 2000 shots:
# 1e-4 for mean((rho - rho_hat) @ (rho - rho_hat)^H)
# 1e-4 for MSE tr((rho - rho_hat) @ (rho - rho_hat)^H)/(d*d)
# 1e-2 for SSE tr((rho - rho_hat) @ (rho - rho_hat)^H)

# For 10000 shots:
# 1e-4
# 1e-4
# 1e-2

# For 100000 shots:
# 1e-4
# 1e-2
# 1e-2

# dump matrix to file 
# mat = Re(t(do.call("cbind",Pra)))
# write.table(round(mat,5), file="Pra_R.txt", row.names=FALSE, col.names=FALSE)
