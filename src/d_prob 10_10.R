library(reticulate)
use_condaenv("road_segmentation")
setwd("/home/daniel/Documents/thesis/src")
source_python("data_generation.py")
source_python("data_generation_exact.py")
seed <- as.integer(0)

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
norm.complex <- function(x) {
   return(x / sqrt(sum(Mod(x)^2)))
}
norm.complex <- compiler::cmpfun(norm.complex)
# taking the diagonal faster #
Di.ag <- function(A, B) {
   return(sapply(1:nrow(A), function(s) A[s, ] %*% B[, s]))
}
Di.ag <- compiler::cmpfun(Di.ag)
# return the eigen vectors
eigen.vec <- function(A) {
   return(eigen(A)$vectors)
}

# Pauli basis 2x2
sx <- matrix(c(0, 1, 1, 0), nr = 2)
sy <- matrix(c(0, 1i, -1i, 0), nr = 2)
sz <- matrix(c(1, 0, 0, -1), nr = 2)
basis <- list(diag(2), sx, sy, sz)

## #qubits
n <- 3
J <- 4^n
I <- 6^n
d <- R <- 2^n
A <- 3^n
## total number of a, b and r
b <- permutations(4, n, repeats.allowed = T)
a <- permutations(3, n, v = c(2, 3, 4), repeats.allowed = T)
r <- permutations(2, n, v = c(-1, 1), repeats.allowed = T)

a_py <- permutations(3, n, v = c(1, 2, 3), repeats.allowed = T)

# return the projectors
projectors <- function(a, r) {
   tem1 <- lapply(basis[a], eigen.vec)
   tem3 <- lapply(1:length(tem1), function(s) {
      return(tem1[[s]][r[s], ])
   })
   # print(tem3)
   # print(tem3[[1]] %*%Conj(t(tem3[[1]])))
   # print("\n")
   lapply(tem3, function(x) x %*% Conj(t(x)))
}

# The projectors matrices
Pra <- list()
count <- 0
for (j in 1:A) {
   for (i in 1:R) {
      count <- count + 1
      # From python
      aj = as.integer(a_py[j,])
      ri = as.integer(r[i,])
      proj <- asplit(projectors_py(aj,ri), MARGIN=1)
      Pra[[count]] = c(kronecker_list(proj))
      # Pra[[count]] <- c(kronecker_list(projectors(a = a[j, ], r = r[i, ])))
   }
}

# The "true-test" dens.ma matrix
## pure state
# dens.ma = matrix(0,nr=d,nc=d)
# dens.ma[1,1] = 1
## mixed state
# u = sapply(1:d, function(i)norm.complex(rcmvnorm(1,sigma=diag(d)/100)))
# dens.ma = Conj(t(u))%*%u/d

# From python
dens.ma <- get_true_rho(as.integer(n), "rank2", seed)

# print(dens.ma)
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
Prob.ar <- matrix(0, nr = A, nc = R)
for (i in 1:A) {
   for (j in 1:R) {
      # From python
      # ai = as.integer(a_py[i,])
      # rj = as.integer(r[j,])
      # proj <- asplit(projectors_py(ai,rj), MARGIN=1)
      # Prob.ar[i,j] <- sum(Di.ag(dens.ma,kronecker_list(proj)))

      Prob.ar[i, j] <- sum(Di.ag(dens.ma, kronecker_list(projectors(a = a[i, ], r = r[j, ]))))
   }
}
Prob.ar <- Re(Prob.ar)

# calculate the probability matrix after simulating sample
n.size <- 2000 ## numbers of repeat the measurements

#  
p_ra = compute_measurements(as.integer(n), dens.ma, as.integer(n.size), seed)

# p_ra <- apply(Prob.ar, 1, function(x) {
#    H <- sample(1:R, n.size, prob = x, replace = TRUE)
#    return(sapply(1:R, function(s) sum(H == s) / n.size))
# })

# transform the matrix to the vector form
p_ra1 <- c(p_ra)
# print(p_ra1)
################################################
####### MAIN CODES: ############################
################################################
# From python
# Pra = get_measurables(as.integer(n))
# Pra = lapply(seq_len(nrow(Pra)), function(i) Pra[i,])
rho <- matrix(0, nr = d, nc = d)

# From python
# Te = random_standard_exponential(as.integer(d), seed)
Te <- rexp(d)

# From python
d_int <- as.integer(d)
U <- random_complex_ortho(d_int, d_int, seed)
# U <- u.hat
# print(U)
Lamb <- c(Te / sum(Te))
ro <- .5
be <- 1

gamm <- n.size / 2
entry <- c()
# MCMC loop
Iter <- 500
burnin <- 100
start_time <- Sys.time()
for (t in 1:(Iter + burnin)) {
   print(t)
   for (j in 1:d) {
      # update Lambda
      Te.can <- Te
      # From python
      # Te.can[j] = Te[j]*exp(be* random_uniform(-0.5, 0.5, as.integer(1), seed))
      Te.can[j] <- Te[j] * exp(be * runif(1, min = -0.5, .5))
      L.can <- Te.can / sum(Te.can)
      tem.can <- c(U %*% diag(L.can) %*% Conj(t(U)))
      tem <- c(U %*% diag(Lamb) %*% Conj(t(U)))
      # Computes the inner product between nu and each combination of (flattened) projectors (results in size 1296, ie 2^n*3^n)
      # and then computes the differences with p_ra1
      ss <- sum((sapply(Pra, function(x) crossprod(tem.can, x)) - p_ra1)^2 -
         (sapply(Pra, function(x) crossprod(tem, x)) - p_ra1)^2)
      r.prior <- (ro - 1) * log(Te.can[j] / Te[j]) - Te.can[j] + Te[j]
      ap <- -gamm * Re(ss)
      # From python
      # if(log(random_uniform(0, 1, as.integer(1), seed)) <= ap + r.prior){Te<- Te.can}
      if (log(runif(1)) <= ap + r.prior) {
         Te <- Te.can
      }
      Lamb <- c(Te / sum(Te))
   }
   # random_multivariate_complex(rep(0, d), diag(d), 1, seed)
   # update U
   for (j in 1:d) {
      U.can <- U
      # From python
      # U.can[, j] <- norm.complex(U[, j] + random_multivariate_complex(rep(0, d), diag(d), as.integer(1), seed) / 100)
      U.can[, j] <- norm.complex(U[, j] + rcmvnorm(1, sigma = diag(d)) / 100)
      tem.can <- c(U.can %*% diag(Lamb) %*% Conj(t(U.can)))
      tem <- c(U %*% diag(Lamb) %*% Conj(t(U)))
      ss <- sum((sapply(Pra, function(x) crossprod(tem.can, x)) - p_ra1)^2 -
         (sapply(Pra, function(x) crossprod(tem, x)) - p_ra1)^2)
      ap <- Re(-gamm * ss)
      # From python
      # if(log(random_uniform(0, 1, as.integer(1), seed)) <= ap){U<- U.can}
      if (log(runif(1)) <= ap) {
         U <- U.can
      }
   }
   # approx rho
   if (t > burnin) {
      rho <- U %*% diag(Lamb) %*% Conj(t(U)) / (t - burnin) + rho * (1 - 1 / (t - burnin))
      #   print(max(Lamb))
   }
}
end_time <- Sys.time()
end_time - start_time
# mean((dens.ma-rho)%*%Conj(t((dens.ma-rho)))) #0.0002687214
MSE <- tr((rho - dens.ma) %*% Conj(t(rho - dens.ma))) / (d * d) # MSE
fro_sq <- tr((rho - dens.ma) %*% Conj(t(rho - dens.ma))) # SSE = fro^2
print(MSE)
print(fro_sq)
