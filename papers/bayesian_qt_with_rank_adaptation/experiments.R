##########################
library(gtools)
library(cmvnorm)
library(ppls)
library(rTensor)
# normalize a complex vector #
norm.complex = function(x) return(x/sqrt(sum(Mod(x)^2)))
# taking the diagonal faster #
Di.ag = function(A,B) return(sapply(1:nrow(A),function(s) A[s,]%*%B[,s]))
# return the eigen vectors
eigen.vec = function(A) return(eigen(A)$vectors)

#Pauli basis 2x2
sx = matrix(c(0,1,1,0),nr=2)
sy = matrix(c(0,1i,-1i,0),nr=2)
sz = matrix(c(1,0,0,-1),nr=2)
basis = list(diag(2),sx,sy,sz)

## #qubits
n = 4
A = 3^n
R = d = 2^n
M = J = 4^n
I = 6^n
## total number of a, b and r
b = permutations(4,n, repeats.allowed=T)
a = permutations(3,n, v=c(2,3,4), repeats.allowed=T)
r = permutations(2,n, v=c(-1,1), repeats.allowed=T)

### Pauli basis for n qubit
sig_b = list()
if (n==1){sig_b = basis
}else{
   for (i in 1:M){
      sig_b[[i]] = kronecker_list(basis[b[i,]])
   }  
}

# return the projectors
projectors = function(a,r){
   tem1 = lapply(basis[a],eigen.vec)
   tem3 = lapply(1:length(tem1), function(s) return(tem1[[s]][r[s],]))
   lapply(tem3, function(x) x%*%Conj(t(x)))
}

# The "true-test" density matrix
## pure state
density = matrix(0,nr=d,nc=d)
density[1,1] = 1
## mixed state
#v1 = norm.complex(rcmvnorm(1,sigma=diag(d)))
#v2 = norm.complex(rcmvnorm(1,sigma=diag(d)))
#density = 0.5*Conj(t(v1))%*%v1 + 0.5*Conj(t(v2))%*%v2

# the probabilities matrix of the P.ar, used to simulate data
Prob.ar = matrix(0,nr=A,nc=R)
if(n==1){
   for(i in 1:A){
      for(j in 1:R){
         Prob.ar[i,j] <- c(density)%*%unlist(projectors(a = a[i,],r = r[j,]))
      }  }
}else{
   for(i in 1:A){
      for(j in 1:R){
         Prob.ar[i,j] <- sum(Di.ag(density,kronecker_list(projectors(a = a[i,],r = r[j,]))))
      }}}
Prob.ar = Re(Prob.ar)

### matrix P_{(ra);b}
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

# calculate the probability matrix after simulating sample
n.size = 100## numbers of repeat the measurements
p_ra = apply(Prob.ar, 1,function(x){
   H = sample(1:R,n.size,prob = x,replace=TRUE)
   return(sapply(1:R,function(s)sum(H==s)/n.size))
})

# transform the matrix to the vector form
p_ra1 = c(p_ra)  
temp1 = p_ra1%*%P_rab
temp1 = temp1/d
# calculating coefficients rho_b in Pauli basis expansion
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

########################################################
####### MAIN CODES: ############################
rho = matrix(0,nr=d,nc=d)
Lamb = c(rdirichlet(1, rep(100000,d)))
   #c((1-alpha)*lamb.hat + alpha*rdirichlet(1, rep(1,d)))
U <- u.hat
MSE = c()
entry = c()
gamm = n.size*A*(9/5)^(n/2)/32

#MCMC loop
Iter = 4000
burnin = 1000
prior = rep(.1,d)
proposal = rep(.1,d) #(1-.9)*lamb.hat + 0.9*rdirichlet(1, rep(100,d))
L.ac = 0
U.ac = 0
bet = 1


for(t in 1:(Iter+burnin)){
   #update Lambda
   L.can = c(rdirichlet(1,proposal))#c((1-alpha)*Lamb + alpha*rdirichlet(1,proposal))
   sum.L = sum(Mod(U%*%diag(L.can)%*%Conj(t(U))-rho.hat)^2
               -Mod(U%*%diag(Lamb)%*%Conj(t(U))-rho.hat)^2)
   pro.L = ddirichlet(Lamb,prior)*ddirichlet(L.can,proposal)
   ap.L = exp(-sum.L*gamm)*ddirichlet(Lamb,proposal)*ddirichlet(L.can,prior)
   if(runif(1) <= ap.L/pro.L){Lamb<- L.can;L.ac = L.ac+1}
   
   #update U
   for(j in 1:d){
      U.can = U
      Unew = norm.complex(U[,j]+bet*rcmvnorm(1,sigma=diag(Lamb)))
      U.can[,j] = Unew
      sum.U = sum(Mod(U.can%*%diag(Lamb)%*%Conj(t(U.can))-rho.hat)^2
                  -Mod(U%*%diag(Lamb)%*%Conj(t(U))-rho.hat)^2)
      ap.U = exp(-gamm*sum.U)
      if(runif(1) <= ap.U){U<- U.can; U.ac = U.ac +1}
      }

   #approx rho
   if(t>burnin){rho = U%*%diag(Lamb)%*%Conj(t(U))/(t-burnin) + rho*(1-1/(t-burnin))
   #MSE[t] =mean((rho.hat-rho)%*%Conj(t((rho.hat-rho))))
   entry[t] = rho[1,1]}
   print(max(Lamb))
}#end MCMC



#Test for the eigeinvalues
eigen(rho)$values
sum(eigen(rho)$values)
sum(diag(rho))
# plot RMSE
xgr = 1:Iter
plot(xgr, MSE[xgr],"l",xlab="",ylab="",col="blue")
#points(RMSE.hat[1:Iter], type="b", pch=20,col="red",ylim = range(0,.5))
abline(h = mean((rho.hat-density)%*%Conj(t((rho.hat-density)))),lwd = 2)

MSE[Iter]
mean(Conj(t((density-rho)))%*%(density-rho))
mean((rho.hat-density)%*%Conj(t((rho.hat-density))))
plot(1:Iter,Re(entry[1:Iter+burnin]),"l")

# possible ideas: 
# 1.heating/cooling the posterior (MC^3)
# 2. mixture of Dirichlet



